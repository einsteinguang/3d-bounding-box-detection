import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

# Local imports (adjust if your code structure differs)
from config.load_param import load_params
from dataloader import Bbox3DDataset, detection_collate, load_single_sample
from modules.transformer_decoder import BBoxDetectionNetLateFusion, BBoxDetectionNetEarlyFusion
from utils.transformation import get_transforms
from utils.loss_function import reconstruct_8_corners_from_6d, estimate_pc_radius
from utils.iou_3d import compute_iou
from utils.preprocessing import compute_bbox_prior
from torch.utils.data import DataLoader


ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


@torch.no_grad()
def plot_pred_and_gt_3d(pc_xyz, gt_bboxes, pred_bboxes, save_path):
    """
    Plots a single sample's point cloud, plus ground-truth bounding boxes
    and predicted bounding boxes. Saves the figure to `save_path`.

    Args:
        gt_bboxes:  list (or np.array) of shape (N, 8, 3), the GT corners
        pred_bboxes:list (or np.array) of shape (N, 8, 3), the predicted corners
        save_path:  path to save the resulting plot
    """

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define color map for different box IDs
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'purple']
    # randomly pick maximum 4 boxes to plot
    if len(gt_bboxes) > 4:
        gt_bboxes = gt_bboxes[:4]
        pred_bboxes = pred_bboxes[:4]

    # Plot GT boxes with solid lines
    for box_id, corners in enumerate(gt_bboxes):
        for (i, j) in [(0,1),(1,2),(2,3),(3,0),
                       (4,5),(5,6),(6,7),(7,4),
                       (0,4),(1,5),(2,6),(3,7)]:
            ax.plot(
                [corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color=colors[box_id % len(colors)]
            )
        ax.scatter(
            pc_xyz[box_id, :, 0],
            pc_xyz[box_id, :, 1],
            pc_xyz[box_id, :, 2],
            c=colors[box_id % len(colors)], alpha=0.1, s=2
        )

    # Plot predicted boxes in another color (red)
    for box_id, corners in enumerate(pred_bboxes):
        for (i, j) in [(0,1),(1,2),(2,3),(3,0),
                       (4,5),(5,6),(6,7),(7,4),
                       (0,4),(1,5),(2,6),(3,7)]:
            ax.plot(
                [corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color=colors[box_id % len(colors)], linestyle='--'
            )

    # Axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ground Truth vs Predicted 3D Boxes")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def run_baseline(params_path, on_training_samples=False, image_name="model_baseline.png"):
    params = load_params(params_path)

    val_dataset = Bbox3DDataset(
        data_path=params.dataset.data_dir,
        params=params,
        train=on_training_samples
    )
    # use this line to also predict the training samples
    if on_training_samples:
        val_dataset.transforms = get_transforms(is_train=False, params=params.dataset.transforms)

    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, collate_fn=detection_collate
    )

    inference_times = []
    with torch.no_grad():
        all_ious = []
        for batch_idx, batch in enumerate(val_loader):
            pc  = [x for x in batch["pc"]]
            bboxes_3d_gt = [x for x in batch["bbox3d"]]

            for i in range(len(pc)):  # loop over batch
                # ground-truth corners => shape (n_gt, 8, 3)
                gt_corners = bboxes_3d_gt[i].cpu().numpy()  # (n_gt, 8, 3)

                # Reconstruct corners for each predicted box
                start_time = time.time()
                pred_corners_list = []
                for j in range(pc[i].shape[0]):  # loop over objects

                    center_prior, dims_prior, orient_6d_prior = compute_bbox_prior(pc[i][j].cpu().numpy())

                    corners_8x3 = reconstruct_8_corners_from_6d(
                        center_prior,
                        dims_prior,
                        orient_6d_prior,
                    )
                    pred_corners_list.append(corners_8x3.unsqueeze(0))
                inference_times.append(time.time() - start_time)

                if len(pred_corners_list) > 0:
                    pred_corners = torch.cat(pred_corners_list, dim=0).cpu().numpy()  # (n_pred,8,3)
                else:
                    pred_corners = np.zeros((0,8,3), dtype=np.float32)

                folder = val_dataset.folders[batch_idx]

                out_path = os.path.join(folder, image_name)

                ious = []
                for b in range(len(gt_corners)):
                    iou = compute_iou(
                        gt_corners[b],  # (8,3)
                        pred_corners[b]   # (8,3)
                    )
                    if iou >= 0. and iou < 1.:
                        ious.append(iou)
                all_ious.extend(ious)

                plot_pred_and_gt_3d(
                    pc[i].cpu().numpy(),          # (n_object, n_sample,3)
                    gt_bboxes = gt_corners,       # (n_gt,8,3)
                    pred_bboxes = pred_corners,   # (n_pred,8,3)
                    save_path = out_path
                )
                print(f"Saved inference result to {out_path}")
        print(f"Average IoU: {np.mean(all_ious):.4f}")
        print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
        return np.mean(all_ious)


def run_inference(params_path, checkpoint_path, on_training_samples=False, image_name="model_early_fusion.png"):
    params = load_params(params_path)

    val_dataset = Bbox3DDataset(
        data_path=params.dataset.data_dir,
        params=params,
        train=on_training_samples
    )
    # use this line to also predict the training samples
    if on_training_samples:
        val_dataset.transforms = get_transforms(is_train=False, params=params.dataset.transforms)

    val_loader = DataLoader(
        val_dataset, batch_size=1,
        shuffle=False, collate_fn=detection_collate
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if hasattr(params.model, 'fusion_style') and params.model.fusion_style == "early_fusion":
        detection_net = BBoxDetectionNetEarlyFusion(params.model, True).to(device)
    else:
        detection_net = BBoxDetectionNetLateFusion(params.model, True).to(device)
    detection_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    detection_net.eval()
    inference_times = []
    with torch.no_grad():
        all_ious = []
        for batch_idx, batch in enumerate(val_loader):
            rgb = batch["rgb"].to(device)      # shape (B,3,H,W)
            pc  = [x.to(device) for x in batch["pc"]]
            msk = [x.to(device) for x in batch["mask"]]
            bboxes_3d_gt = [x for x in batch["bbox3d"]]

            start_time = time.time()
            preds = detection_net(rgb, pc, msk)  # also a list of length B
            inference_times.append(time.time() - start_time)

            for i in range(len(preds)):  # loop over batch
                # ground-truth corners => shape (n_gt, 8, 3)
                gt_corners = bboxes_3d_gt[i].cpu().numpy()  # (n_gt, 8, 3)

                # predicted parameters => shape (n_pred, 12) = center(3)+dims(3)+orient_6d(6)
                pred_params = preds[i]  # (n_pred, 12)

                # Reconstruct corners for each predicted box
                pred_corners_list = []
                for j in range(pred_params.shape[0]):  # loop over objects
                    # parse out (center, dims, orient_6d)
                    center_3 = pred_params[j, 0:3]
                    dims_3 = pred_params[j, 3:6]
                    orient_6d = pred_params[j, 6:12]

                    center_prior, dims_prior, orient_6d_prior = compute_bbox_prior(pc[i][j].cpu().numpy(), device=device)

                    corners_8x3 = reconstruct_8_corners_from_6d(
                        center_prior + center_3,
                        dims_prior + dims_3,
                        orient_6d_prior + orient_6d,
                    )
                    pred_corners_list.append(corners_8x3.unsqueeze(0))
                if len(pred_corners_list) > 0:
                    pred_corners = torch.cat(pred_corners_list, dim=0).cpu().numpy()  # (n_pred,8,3)
                else:
                    pred_corners = np.zeros((0,8,3), dtype=np.float32)

                folder = val_dataset.folders[batch_idx]

                out_path = os.path.join(folder, image_name)

                ious = []
                for b in range(len(gt_corners)):
                    iou = compute_iou(
                        gt_corners[b],  # (8,3)
                        pred_corners[b]   # (8,3)
                    )
                    if iou >= 0. and iou < 1.:
                        ious.append(iou)
                all_ious.extend(ious)

                plot_pred_and_gt_3d(
                    pc[i].cpu().numpy(),          # (n_object, n_sample,3)
                    gt_bboxes = gt_corners,       # (n_gt,8,3)
                    pred_bboxes = pred_corners,   # (n_pred,8,3)
                    save_path = out_path
                )
                print(f"Saved inference result to {out_path}")
        print(f"Average IoU: {np.mean(all_ious):.4f}")
        print(f"Average inference time: {np.mean(inference_times):.4f} seconds")
        return np.mean(all_ious)


if __name__ == "__main__":
    run_baseline(f"config/params_early_fusion.yaml", on_training_samples=False,)
    run_inference(f"config/params_early_fusion.yaml",
                  f"checkpoints/model_early_fusion.pth",
                  on_training_samples=False,
                  image_name="model_early_fusion.png")
    run_inference(f"config/params_late_fusion.yaml",
                f"checkpoints/model_late_fusion.pth",
                on_training_samples=False,
                image_name="model_late_fusion.png")
