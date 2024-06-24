import numpy as np
from PIL import Image

path = (
    "/data/teton_data/train/nfs/1420122041549_3/2024_01_31/2024_01_31_14_45_59_262_0/"
)


if __name__ == "__main__":
    # Load image
    z = Image.open(path + "depth/0.png")
    z = np.array(z)

    # Load camera intrinsics
    K = np.array(
        [
            [525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0],
        ]
    )

    # Extrinsics
    R = np.array(
        [
            [0.9998, 0.0, 0.0186],
            [0.0, 1.0, 0.0],
            [-0.0186, 0.0, 0.9998],
        ]
    )

    t = [0.0, 0.0, 0.0]

    Rt = np.hstack((R, np.array(t).reshape(3, 1)))

    P = K @ Rt

    # Compute 3D points

    # 2D locations = K @ T @ 3D points
    # We want to solve for 3D points
    # 3D points = (K @ T)^-1 @ 2D locations

    # Create mesh of coordinates including Z
    x = np.arange(0, z.shape[1])
    y = np.arange(0, z.shape[0])
    xx, yy = np.meshgrid(x, y)

    coords = np.stack([xx * z, yy * z, z, np.ones_like(z)], axis=-1)

    a = np.vstack(
        (
            np.hstack((K, np.zeros((3, 1)))),
            np.array([0, 0, 0, 1]),
        )
    )

    b = np.vstack((Rt, np.array([0, 0, 0, 1])))

    T = np.linalg.inv(a @ b)

    # [u, v, 1, 1/z] = 1/z * [K, 0; 0, 1] @ [Rt, 0; 0, 1] @ [X, Y, Z, 1]
    # Compute 3D points

    points = np.linalg.inv(T) @ coords.reshape(-1, 4).T

    points = points.T[:, :3].reshape(z.shape[0], z.shape[1], 3)

    print(points.shape)
