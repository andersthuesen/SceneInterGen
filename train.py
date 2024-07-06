import os
import time
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models.intergen import InterGen
from models.utils import CosineWarmupScheduler
from utils.utils import print_current_loss

from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
from lightning.pytorch.strategies import DDPStrategy

torch.set_float32_matmul_precision("medium")

import smplx


class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, "model")
        self.meta_dir = pjoin(self.save_root, "meta")
        self.log_dir = pjoin(self.save_root, "log")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

    def _configure_optim(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.TRAIN.LR),
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            fused=True,
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True
        )
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch: dict):
        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss, "loss_logs": loss_logs}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(outputs["loss_logs"], on_step=True, on_epoch=False, prog_bar=True)

        # if outputs.get("skip_batch") or not outputs.get("loss_logs"):
        #     return

        # if self.global_step % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
        #     print_current_loss(
        #         self.start_time,
        #         self.global_step,
        #         outputs["loss_logs"],
        #         self.trainer.current_epoch,
        #         inner_iter=batch_idx,
        #         lr=self.trainer.optimizers[0].param_groups[0]["lr"],
        #     )

    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    # def save(self, file_name):
    #     state = {}
    #     try:
    #         state["model"] = self.model.module.state_dict()
    #     except:
    #         state["model"] = self.model.state_dict()
    #     torch.save(state, file_name, _use_new_zipfile_serialization=False)
    #     return


if __name__ == "__main__":
    model_cfg = get_config("configs/model.yaml")
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml")

    smpl = smplx.SMPLLayer(
        model_path=model_cfg.SMPL_MODEL_PATH,
    )
    mean = torch.load("mean.pt")
    std = torch.load("std.pt")

    datamodule = DataModule(
        data_cfg,
        train_cfg.TRAIN.BATCH_SIZE,
        train_cfg.TRAIN.NUM_WORKERS,
        smpl,
        mean,
        std,
    )

    model = InterGen(model_cfg, mean, std)

    # Remove after use
    # ckpt = torch.load(
    #     "checkpoints/IG-S-8/model/epoch=499-step=80500.ckpt", map_location="cpu"
    # )
    # for k in list(ckpt["state_dict"].keys()):
    #     if "model" in k:
    #         ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
    # model.load_state_dict(ckpt["state_dict"], strict=True)
    # print("checkpoint state loaded!")

    litmodel = LitTrainModel(model, train_cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=litmodel.model_dir, every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH
    )
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices="auto",
        accelerator="gpu",
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(save_dir=litmodel.log_dir),
        log_every_n_steps=train_cfg.TRAIN.LOG_STEPS,
    )

    trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=train_cfg.TRAIN.RESUME)
