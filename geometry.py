from typing import Optional
import torch


# From Eq. 3 in https://arxiv.org/pdf/2202.12555.pdf
def rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    a1, a2 = rot6d.chunk(2, dim=-1)

    b1 = a1 / torch.norm(a1, dim=-2, keepdim=True)
    u2 = a2 - (b1 * torch.einsum("...ij,...ij->...j", a2, b1).unsqueeze(-1))
    b2 = u2 / torch.norm(u2, dim=-2, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-2)

    rotmat = torch.cat((b1, b2, b3), dim=-1)
    return rotmat


def rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    # Select first two columns of rotation matrix
    return rotmat[..., :2]


def perspective_projection(
    points: torch.Tensor,
    translation: torch.Tensor,
    focal_length: torch.Tensor,
    camera_center: Optional[torch.Tensor] = None,
    rotation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = (
            torch.eye(3, device=points.device, dtype=points.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
    if camera_center is None:
        camera_center = torch.zeros(
            batch_size, 2, device=points.device, dtype=points.dtype
        )
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    return projected_points[..., :-1]
