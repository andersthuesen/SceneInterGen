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

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
from lightning.pytorch.strategies import DDPStrategy

torch.set_float32_matmul_precision("medium")

import smplx


class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, "model")
        self.meta_dir = pjoin(self.save_root, "meta")
        self.log_dir = pjoin(self.save_root, "log")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.writer = SummaryWriter(self.log_dir)

    def _configure_optim(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.TRAIN.LR),
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True
        )
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        (
            motion,
            motion_mask,
            classes,
            actions,
            object_points,
            object_points_mask,
            description_tokens,
            description_embs,
        ) = batch_data

        batch = OrderedDict(
            {
                "motion": motion,
                "motion_mask": motion_mask,
                "classes": classes,
                "actions": actions,
                "object_points": object_points,
                "object_points_mask": object_points_mask,
                "description_tokens": description_tokens,
                "description_embs": description_embs,
            }
        )

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()

    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss, "loss_logs": loss_logs}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get("skip_batch") or not outputs.get("loss_logs"):
            return
        for k, v in outputs["loss_logs"].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            print_current_loss(
                self.start_time,
                self.it,
                mean_loss,
                self.trainer.current_epoch,
                inner_iter=batch_idx,
                lr=self.trainer.optimizers[0].param_groups[0]["lr"],
            )

    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def save(self, file_name):
        state = {}
        try:
            state["model"] = self.model.module.state_dict()
        except:
            state["model"] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


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
    )

    trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=train_cfg.TRAIN.RESUME)
