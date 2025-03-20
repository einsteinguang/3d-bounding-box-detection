import os
import sys
import glob
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch

from .loss_function import reconstruct_8_corners_from_6d


ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# RGB shape: (h, w, 3), dtype: uint8
# Mask shape: (n, h, w), dtype: bool, n is number of masks, corresponding to number of objects in the image
# Point Cloud shape: (3, h, w), dtype: float64, h and w are same as the RGB image

# label:
# Bbox3D shape: (n, 8, 3), dtype: float32


def load_single_sample(folder_path):
    sample = {}
    # Load RGB image
    rgb_path = os.path.join(folder_path, 'rgb.jpg')
    sample['rgb'] = np.array(Image.open(rgb_path))
    sample['bbox3d'] = np.load(os.path.join(folder_path, 'bbox3d.npy'))
    sample['mask'] = np.load(os.path.join(folder_path, 'mask.npy'))
    sample['pc'] = np.load(os.path.join(folder_path, 'pc.npy'))
    return sample


def get_all_sample_folders(root_dir):
    """
    Get list of all sample folders in the root directory

    Args:
        root_dir (str): Path to the root directory containing all samples

    Returns:
        list: List of folder paths
    """
    # Get all subdirectories
    folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    # sort the folders by the folder name
    folders.sort()
    return folders


def inspect_dataset(root_dir, num_samples=500):
    """
    Inspect the dataset by loading and printing information about several samples

    Args:
        root_dir (str): Path to the root directory containing all samples
        num_samples (int): Number of samples to inspect
    """
    folders = get_all_sample_folders(root_dir)
    print(f"Total number of samples: {len(folders)}")

    max_h, max_w = 0, 0

    # Inspect a few samples
    for i in range(min(num_samples, len(folders))):

        sample = load_single_sample(folders[i])

        if sample['mask'].shape[0] != sample['bbox3d'].shape[0]:
            print(f"Sample {i+1} has mismatched mask and bbox3d shapes: {sample['mask'].shape[0]} vs {sample['bbox3d'].shape[0]}")

        # check the point cloud dimension and RGB image dimension
        if sample['pc'].shape[1] != sample['rgb'].shape[0] or sample['pc'].shape[2] != sample['rgb'].shape[1]:
            print(f"Sample {i+1} has mismatched pc and rgb shapes: {sample['pc'].shape} vs {sample['rgb'].shape}")

        # update max_h and max_w
        max_h = max(max_h, sample['rgb'].shape[0])
        max_w = max(max_w, sample['rgb'].shape[1])

        # # Print shapes and data types
        # print(f"RGB shape: {sample['rgb'].shape}, dtype: {sample['rgb'].dtype}")
        # print(f"Bbox3D shape: {sample['bbox3d'].shape}, dtype: {sample['bbox3d'].dtype}")
        # print(f"Mask shape: {sample['mask'].shape}, dtype: {sample['mask'].dtype}")
        # print(f"Point Cloud shape: {sample['pc'].shape}, dtype: {sample['pc'].dtype}")

        # # Print some sample values
        # print("\nSample values:")
        # print(f"Bbox3D first row: {sample['bbox3d'][0]}")
        # print(f"Point Cloud first point: {sample['pc'][0]}")
        # print(f"Mask value range: [{sample['mask'].min()}, {sample['mask'].max()}]")
    print(f"Max height: {max_h}, Max width: {max_w}")


def compute_bbox_prior(sampled, coverage=98, device=torch.device('cpu')):
    """
    Compute a bounding box prior from a masked point cloud.

    Args:
        pc_points (np.ndarray): Array of shape (num_points, 3) with (x, y, z) coordinates.
        coverage (float): Percentage of points to cover (e.g. 90 for 5th to 95th percentile).
        n_samples (int): Maximum number of points to sample.

    Returns:
        bbox_center (torch.Tensor): (3,) center of the box.
        dims (torch.Tensor): (3,) dimensions (l, w, h) ordered descending.
        orient_6d (torch.Tensor): (6,) orientation, defined as the concatenation of the
                                  two principal axes corresponding to the largest and second largest dims.
    """
    # Center the points and compute PCA using SVD.
    mean = np.mean(sampled, axis=0)
    centered = sampled - mean  # (n, 3)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Columns of eigenvectors are in descending order of variance.
    eigenvectors = Vt.T  # shape (3, 3)

    # Project points onto each principal axis.
    projections = np.dot(centered, eigenvectors)  # shape (n, 3)

    # For each axis compute the 5th and 95th percentiles.
    lower = np.percentile(projections, 100-coverage, axis=0)  # shape (3,)
    upper = np.percentile(projections, coverage, axis=0) # shape (3,)
    dims = upper - lower                         # raw dims per axis
    center_pca = (upper + lower) / 2.0             # center in PCA coordinate space

    # Order the dimensions descending (largest first) and reorder the eigenvectors and center_pca accordingly.
    order = np.argsort(dims)[::-1]  # indices sorted in descending order
    dims = dims[order]
    center_pca = center_pca[order]
    eigenvectors = eigenvectors[:, order]  # reorder columns accordingly

    # Build orient_6d from the first two principal axes.
    orient_6d = np.concatenate([eigenvectors[:, 0], eigenvectors[:, 1]], axis=0)  # shape (6,)

    # Compute the bounding box center in original coordinates.
    # The offset is the dot product of the center in PCA space with the eigenvectors.
    offset = np.dot(center_pca, eigenvectors.T)  # (3,)
    bbox_center = mean + offset  # (3,)
    # if dims only has 1 or 2 dimensions, pad with 0.01
    if len(dims) < 3:
        dims = np.concatenate([dims, np.array([0.01] * (3 - len(dims)))])

    # Convert to torch tensors.
    bbox_center = torch.tensor(bbox_center, dtype=torch.float, device=device)
    dims = torch.tensor(dims, dtype=torch.float, device=device)
    orient_6d = torch.tensor(orient_6d, dtype=torch.float, device=device)

    return bbox_center, dims, orient_6d


def plot_bbox_and_pointcloud(folder_path):
    """
    Plot 3D bounding boxes and point cloud, and save the plot to the folder

    Args:
        folder_path (str): Path to the sample folder
    """
    import matplotlib.pyplot as plt

    # Load the sample
    sample = load_single_sample(folder_path)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot point cloud (downsample for visualization)
    pc = sample['pc']
    stride = 10  # Adjust this value to change point density
    # ax.scatter(pc[0, ::stride, ::stride],
    #             pc[1, ::stride, ::stride],
    #             pc[2, ::stride, ::stride],
    #             c='black', alpha=0.2, s=1)

    # Plot each bounding box
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'purple']
    for b, bbox in enumerate(sample['bbox3d']):
        mask = sample['mask'][b]
        # plot the point cloud for the mask
        pc_mask = pc[:, mask]
        # warning if pc_mask is empty
        if pc_mask.shape[1] == 0:
            print(f"Mask {b} is empty")
            continue
        color = colors[b % len(colors)]
        ax.scatter(pc_mask[0,::stride], pc_mask[1,::stride], pc_mask[2,::stride], color=color, alpha=0.2, s=1)
        # transpose the pc_mask to (n, 3) for compute_bbox_prior
        pc_points = pc_mask.transpose(1, 0)
        t1 = time.time()
        n_samples = 5000
        num_points = pc_points.shape[0]
        if num_points > n_samples:
            idx = np.random.choice(num_points, n_samples, replace=False)
            sampled = pc_points[idx]
        else:
            sampled = pc_points
        bbox_center, dims, orient_6d = compute_bbox_prior(sampled)
        print(f"Time taken to compute bbox prior: {time.time() - t1:.3f} seconds")
        corners = reconstruct_8_corners_from_6d(bbox_center, dims, orient_6d)
        # Plot the 3D bounding box
        for i, j in [(0,1), (1,2), (2,3), (3,0),  # Bottom face
                        (4,5), (5,6), (6,7), (7,4),  # Top face
                        (0,4), (1,5), (2,6), (3,7)]: # Vertical edges
            ax.plot3D([corners[i,0], corners[j,0]],
                        [corners[i,1], corners[j,1]],
                        [corners[i,2], corners[j,2]], color=color, linestyle='--')
        # # Plot lines connecting the 8 points
        # for i, j in [(0,1), (1,2), (2,3), (3,0),  # Bottom face
        #                 (4,5), (5,6), (6,7), (7,4),  # Top face
        #                 (0,4), (1,5), (2,6), (3,7)]: # Vertical edges
        #     ax.plot3D([bbox[i,0], bbox[j,0]],
        #                 [bbox[i,1], bbox[j,1]],
        #                 [bbox[i,2], bbox[j,2]], color=color)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Bounding Boxes and Point Cloud')

    # Save plot
    plt.savefig(os.path.join(folder_path, 'pc_objects.png'))
    plt.close()


def down_sample_mask(folder_path):
    """
    Downscale masks by factors of 16 and 32 using nearest neighbor approach
    and save them in the folder.

    Args:
        folder_path (str): Path to the sample folder
    """
    sample = load_single_sample(folder_path)
    masks = sample['mask']  # shape (n, h, w)

    # Downsample by 16
    h16 = masks.shape[1] // 16
    w16 = masks.shape[2] // 16
    masks_16 = np.zeros((masks.shape[0], h16, w16), dtype=bool)
    for i in range(masks.shape[0]):
        masks_16[i] = cv2.resize(masks[i].astype(np.uint8), (w16, h16),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

    # Downsample by 32
    h32 = masks.shape[1] // 32
    w32 = masks.shape[2] // 32
    masks_32 = np.zeros((masks.shape[0], h32, w32), dtype=bool)
    for i in range(masks.shape[0]):
        masks_32[i] = cv2.resize(masks[i].astype(np.uint8), (w32, h32),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

    # Save the downsampled masks
    # Save masks_16 as binary images
    for i in range(masks_16.shape[0]):
        mask_img = (masks_16[i] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder_path, f'mask_16_{i}.png'), mask_img)

    # Save masks_32 as binary images
    for i in range(masks_32.shape[0]):
        mask_img = (masks_32[i] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder_path, f'mask_32_{i}.png'), mask_img)


def plot_bbox_on_image(folder_path):
    """
    Loads an RGB image, its point cloud (camera-aligned),
    and 3D boxes. Then projects the 3D boxes into the image
    and draws them in 2D.
    Saves the result as 'bbox3d_on_image.png' in the folder.
    """
    sample = load_single_sample(folder_path)
    rgb = sample["rgb"]          # shape (h, w, 3) dtype=uint8
    pc  = sample["pc"]           # shape (3, h, w)
    bboxes_3d = sample["bbox3d"] # shape (n, 8, 3)

    # Dimensions of the image
    h, w, _ = rgb.shape

    # Get their corresponding 3D points from pc:
    # pc is shaped (3, h, w), so pc[:, row, col]
    p_tl = pc[:, 0,       0]       # top-left
    p_tr = pc[:, 0,       w - 1]   # top-right
    p_bl = pc[:, h - 1,   0]       # bottom-left
    p_br = pc[:, h - 1,   w - 1]   # bottom-right

    X_tl, Y_tl, Z_tl = p_tl
    X_tr, Y_tr, Z_tr = p_tr
    X_bl, Y_bl, Z_bl = p_bl
    # bottom-right is a consistency check if you want

    # handle any potential divisions by zero
    eps = 1e-12
    denom_x = (X_tr / (Z_tr + eps)) - (X_tl / (Z_tl + eps))
    denom_y = (Y_bl / (Z_bl + eps)) - (Y_tl / (Z_tl + eps))

    if abs(denom_x) < 1e-9 or abs(denom_y) < 1e-9:
        print("Warning: cannot solve intrinsics from corners reliably.")
        return

    fx = (w - 1) / denom_x
    cx = -fx * (X_tl / (Z_tl + eps))

    fy = (h - 1) / denom_y
    cy = -fy * (Y_tl / (Z_tl + eps))

    def project_pts_3d_to_2d(pts_3d):
        zs = pts_3d[..., 2] + eps
        xs = pts_3d[..., 0]
        ys = pts_3d[..., 1]

        us = fx * (xs / zs) + cx
        vs = fy * (ys / zs) + cy
        return np.stack([us, vs], axis=-1)

    # ------------------------------------------------------
    # 3) For each bounding box, project corners and draw lines
    # ------------------------------------------------------
    # We'll create a draw-able PIL image
    pil_img = Image.fromarray(rgb)  # shape (h, w, 3)
    draw = ImageDraw.Draw(pil_img)

    # The 12 edges of the (8-corner) box:
    box_edges = [
        (0,1),(1,2),(2,3),(3,0),      # top face
        (4,5),(5,6),(6,7),(7,4),      # bottom face
        (0,4),(1,5),(2,6),(3,7)       # vertical edges
    ]

    # pick a few colors if you want to vary them per box
    colors = ["red","green","blue","yellow","magenta","cyan","orange","white"]

    for box_idx, corners_8x3 in enumerate(bboxes_3d):
        corners_2d = project_pts_3d_to_2d(corners_8x3)  # shape (8,2)
        c = colors[box_idx % len(colors)]

        # Draw the edges
        for (i, j) in box_edges:
            x1, y1 = corners_2d[i]
            x2, y2 = corners_2d[j]
            draw.line([(x1, y1), (x2, y2)], fill=c, width=2)

    # Save the result
    out_path = os.path.join(folder_path, "bbox3d_on_image.png")
    pil_img.save(out_path)
    print(f"Saved 3D box overlay to {out_path}")


if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = ROOT_DIR + "/../dl_challenge/"

    folders = get_all_sample_folders(dataset_path)
    # print(folders)

    for folder in folders:
        plot_bbox_and_pointcloud(folder)
        # plot_bbox_on_image(folder)
        # down_sample_mask(folder)

    # Inspect the dataset
    # inspect_dataset(dataset_path)