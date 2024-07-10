from os.path import join as pjoin
import click
import torch

from geometry import rot6d_to_rotmat, rotmat_to_rot6d

from tqdm import tqdm

import smplx


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--device", default="cpu")
@click.option("--num-steps", default=1000)
def cli(path: str, device: str, num_steps: int):
    device = torch.device(device)

    smpl = smplx.SMPLLayer(
        model_path="weights/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    ).to(device)

    joints_path = pjoin(path, "joints.pt")
    pose_path = pjoin(path, "poses.pt")

    joints = torch.load(joints_path, map_location=device)[0]
    body_pose = torch.load(pose_path, map_location=device)[0]

    num_people, num_frames = joints.shape[:2]

    # Pick the root joint
    translation = joints[..., 0, :].requires_grad_()
    global_orient_6d = (
        rotmat_to_rot6d(torch.eye(3, device=device))[None, None]
        .repeat(num_people, num_frames, 1, 1)
        .requires_grad_()
    )
    body_pose_6d = rotmat_to_rot6d(body_pose)

    opt = torch.optim.Adam([translation, global_orient_6d, body_pose_6d], lr=0.1)

    steps = tqdm(range(num_steps), desc="Generating motion")
    for step in steps:
        opt.zero_grad()

        global_orient = rot6d_to_rotmat(global_orient_6d)
        body_pose = rot6d_to_rotmat(body_pose_6d)

        smpl_out = smpl(
            global_orient=global_orient.view(-1, 3, 3),
            body_pose=body_pose.view(-1, 23, 3, 3),
            transl=translation.view(-1, 3),
        )

        # Get smpl joints
        smpl_joints = smpl_out.joints.view(num_people, num_frames, -1, 3)[:, :, :24]

        loss = torch.nn.functional.mse_loss(smpl_joints, joints)
        steps.set_postfix(loss=loss.item())

        loss.backward()

        opt.step()

    motion = torch.cat(
        (
            torch.ones_like(translation[..., :1]),
            translation,
            global_orient.flatten(start_dim=-2),
            body_pose.flatten(start_dim=-3),
        ),
        dim=-1,
    )
    motion_path = pjoin(path, "motion.pt")
    print(f"Saving motion to {motion_path}")
    torch.save(motion.detach().cpu().clone(), motion_path)


if __name__ == "__main__":
    cli()
