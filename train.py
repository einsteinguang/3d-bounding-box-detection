import os
import sys
import torch
import mlflow
import numpy as np
import torch.optim as optim
from box import ConfigBox
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config.load_param import load_params
from dataloader import Bbox3DDataset, detection_collate
from modules.transformer_decoder import BBoxDetectionNetLateFusion, BBoxDetectionNetEarlyFusion
from utils.loss_function import compute_loss, extract_6d_ordered_params_from_corners, \
    extract_6d_params_from_corners, reconstruct_8_corners_from_6d
from utils.iou_3d import compute_iou
from utils.preprocessing import compute_bbox_prior


ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def train(params: ConfigBox):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    print("Local gpu rank: ", local_rank)

    data_path = params.dataset.data_dir  # e.g. "dl_challenge"
    batch_size = params.training.batch_size
    num_epochs = params.training.epochs
    lr = params.training.lr
    weight_decay = params.training.weight_decay
    weight_center = params.training.weight_center
    weight_dims = params.training.weight_dims
    weight_orient = params.training.weight_orient

    # 1) Create train & val datasets
    train_dataset = Bbox3DDataset(data_path, params, train=True)
    val_dataset = Bbox3DDataset(data_path, params, train=False)
    sampler = DistributedSampler(train_dataset, shuffle=True,
                                 num_replicas=torch.distributed.get_world_size(),
                                 rank=torch.distributed.get_rank())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=detection_collate
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=detection_collate, shuffle=False
    )

    # Create model
    if hasattr(params.model, 'fusion_style') and params.model.fusion_style == "early_fusion":
        detection_net = BBoxDetectionNetEarlyFusion(params.model, True).to(device)
    else:
        detection_net = BBoxDetectionNetLateFusion(params.model, True).to(device)
    detection_net = DDP(detection_net, device_ids=[local_rank], find_unused_parameters=True)

    # Create parameter groups with different learning rates
    backbone_params = []
    other_params = []
    for name, param in detection_net.module.named_parameters():
        if 'img_encoder.backbone' in name or 'img_encoder.layer' in name:
            # if local_rank == 0:
            #     print(f"Adding {name} to backbone params")
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': params.training.lr_img_backbone},
        {'params': other_params, 'lr': params.training.lr}
    ], weight_decay=weight_decay)

    # LR scheduler if desired
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * num_epochs
    )

    # Training loop
    train_loss = []
    val_loss = []
    iou_curve = []
    epoch_iter = tqdm(range(num_epochs), desc=f"Training progress ", disable=local_rank != 0)
    for epoch in epoch_iter:
        detection_net.train()  # set model to training mode
        count = 0
        epoch_loss = []
        for batch in train_loader:
            rgb = batch["rgb"].to(device)      # (B,3,H,W)
            pc = [x.to(device) for x in batch["pc"]]
            mask = [x.to(device) for x in batch["mask"]]
            bboxes3d = [x.to(device) for x in batch["bbox3d"]]

            preds = detection_net(rgb, pc, mask)

            # Compute loss
            loss_center = []
            loss_dims = []
            loss_orient = []
            for i, gt_boxes in enumerate(bboxes3d):
                # gt_boxes: (N_i, 8, 3)
                pred_boxes = preds[i]
                for j in range(gt_boxes.shape[0]):
                    gt_corners_8x3 = gt_boxes[j]
                    gt_center, gt_dims, gt_orient6d = extract_6d_ordered_params_from_corners(gt_corners_8x3)
                    center, dims, orient_6d = compute_bbox_prior(pc[i][j].cpu().numpy(), device=device)
                    loss_ij = compute_loss(
                        pred_boxes[j], gt_center - center, gt_dims - dims, gt_orient6d - orient_6d)
                    loss_center.append(loss_ij["center"])
                    loss_dims.append(loss_ij["dims"])
                    loss_orient.append(loss_ij["orient"])
            sample_losses = [weight_center * torch.stack(loss_center).mean(),
                             weight_dims * torch.stack(loss_dims).mean(),
                             weight_orient * torch.stack(loss_orient).mean()]
            # print(f"Center loss: {sample_losses[0].item():.4f}, "
            #       f"Dim loss: {sample_losses[1].item():.4f}, "
            #       f"Orient loss: {sample_losses[2].item():.4f}")
            loss = sample_losses[0] + sample_losses[1] + sample_losses[2]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            count += 1
            epoch_loss.append(loss.item())

        train_loss.append([sum(epoch_loss) / count, epoch])
        epoch_iter.set_postfix({"train_loss": train_loss[-1][0]})
        # Validation step (optional IoU checking)
        if (epoch + 1) % params.training.eval_interval == 0 and local_rank == 0:
            detection_net.eval()
            with torch.no_grad():
                ious = []
                loss_center = []
                loss_dims = []
                loss_orient = []
                for batch in val_loader:
                    rgb = batch["rgb"].to(device)
                    pc = [x.to(device) for x in batch["pc"]]
                    mask = [x.to(device) for x in batch["mask"]]
                    bboxes3d = [x.to(device) for x in batch["bbox3d"]]

                    preds = detection_net(rgb, pc, mask)
                    for i, gt_boxes in enumerate(bboxes3d):
                        pred_boxes = preds[i]
                        for j in range(gt_boxes.shape[0]):
                            gt_corners_8x3 = gt_boxes[j]
                            gt_center, gt_dims, gt_orient6d = extract_6d_ordered_params_from_corners(gt_corners_8x3)

                            center, dims, orient_6d = compute_bbox_prior(pc[i][j].cpu().numpy(), device=device)

                            loss_ij = compute_loss(
                                pred_boxes[j], gt_center - center, gt_dims - dims, gt_orient6d - orient_6d)
                            loss_center.append(loss_ij["center"])
                            loss_dims.append(loss_ij["dims"])
                            loss_orient.append(loss_ij["orient"])

                            reconstructed_8_points = reconstruct_8_corners_from_6d(
                                pred_boxes[j][:3] + center, pred_boxes[j][3:6] + dims, pred_boxes[j][6:] + orient_6d
                            )
                            iou = compute_iou(gt_corners_8x3.cpu().numpy(),
                                              reconstructed_8_points.cpu().numpy())
                            ious.append(iou)

                loss = weight_center * torch.stack(loss_center).mean() + \
                       weight_dims * torch.stack(loss_dims).mean() + \
                       weight_orient * torch.stack(loss_orient).mean()
                # print(f"Epoch {epoch} validation loss: {loss.item():.4f}")
                val_loss.append([loss.item(), epoch])

                # print(f"Avg IoU: {np.mean(ious):.4f}")
                iou_curve.append([np.mean(ious), epoch])

        # save model at checkpoint_interval
        if (epoch + 1) % params.training.checkpoint_interval == 0 and local_rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/epoch_{epoch}_{params.model.fusion_style}.pth"
            torch.save(detection_net.module.state_dict(), checkpoint_path)
            # print(f"Model saved at epoch {epoch}")

        # save training and validation loss as np array for later plots
        if local_rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            np.save(f"checkpoints/train_loss_{params.model.fusion_style}.npy", np.array(train_loss))
            np.save(f"checkpoints/val_loss_{params.model.fusion_style}.npy", np.array(val_loss))
            np.save(f"checkpoints/iou_curve_{params.model.fusion_style}.npy", np.array(iou_curve))
    print("Training complete.")


if __name__ == "__main__":
    params_path = "config/params_late_fusion.yaml"
    params = load_params(params_path)
    train(params)