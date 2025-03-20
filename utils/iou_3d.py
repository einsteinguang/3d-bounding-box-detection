import os
import sys
import torch
import time
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

from .loss_function import extract_6d_params_from_corners


ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def compute_iou(box1_vertices, box2_vertices, n_samples=100000):
    """
    Approximate IoU by uniformly sampling random points in the bounding
    region containing both boxes. Vectorized version for faster execution.
    """
    # Convert corners to NumPy
    box1_vertices = np.asarray(box1_vertices)
    box2_vertices = np.asarray(box2_vertices)

    # Axis-aligned bounds
    all_pts = np.vstack([box1_vertices, box2_vertices])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)

    # Extract the box parameters (center, dims, 6D orientation)
    c1, d1, o6d1 = extract_6d_params_from_corners(torch.from_numpy(box1_vertices))
    c2, d2, o6d2 = extract_6d_params_from_corners(torch.from_numpy(box2_vertices))

    # Convert to NumPy
    c1, d1, o6d1 = c1.numpy(), d1.numpy(), o6d1.numpy()
    c2, d2, o6d2 = c2.numpy(), d2.numpy(), o6d2.numpy()

    # Recover b1,b2,b3 from o6d
    def get_axes(o6d):
        b1 = o6d[:3]
        b2 = o6d[3:]
        b1 = b1 / (np.linalg.norm(b1) + 1e-8)
        # Orthogonalize b2
        b2 = b2 - b1 * (b2.dot(b1))
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        return b1, b2, b3

    b1_1, b2_1, b3_1 = get_axes(o6d1)
    b1_2, b2_2, b3_2 = get_axes(o6d2)

    # Sample random points
    rand_xyz = np.random.rand(n_samples, 3) * (maxs - mins) + mins

    def in_box(points, center, dims, b1, b2, b3):
        local = points - center  # shape (n_samples, 3)
        # dot along axis=1 => shape (n_samples,)
        x = np.einsum('ij,j->i', local, b1)
        y = np.einsum('ij,j->i', local, b2)
        z = np.einsum('ij,j->i', local, b3)
        half_x, half_y, half_z = dims / 2.0
        inside = (
            (np.abs(x) <= half_x) &
            (np.abs(y) <= half_y) &
            (np.abs(z) <= half_z)
        )
        return inside

    inside1 = in_box(rand_xyz, c1, d1, b1_1, b2_1, b3_1)
    inside2 = in_box(rand_xyz, c2, d2, b1_2, b2_2, b3_2)

    intersection = np.count_nonzero(inside1 & inside2)
    union = np.count_nonzero(inside1 | inside2)

    if union == 0:
        return 0.0
    return intersection / float(union)


def test_iou():
    root_path = ROOT_DIR + "/../dl_challenge/data_181/"
    boxes_gt = np.load(root_path + "boxes_gt.npy")
    boxes_pred = np.load(root_path + "boxes_pred.npy")

    ious = []
    for i in range(7):
        box1_vertices = boxes_gt[i]
        box2_vertices = boxes_pred[i]
        t1 = time.time()
        iou = compute_iou(box1_vertices, box2_vertices)
        ious.append(iou)
        print(f"Approximate IoU: {iou} (computed in {time.time() - t1:.4f} seconds)")
    print(f"Average IoU: {np.mean(ious):.4f}")


if __name__ == "__main__":
    test_iou()