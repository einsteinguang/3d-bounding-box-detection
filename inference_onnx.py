import onnxruntime
import numpy as np
import torch
import time
import sys
import os
import argparse
from config.load_param import load_params
from dataloader import Bbox3DDataset, detection_collate
from utils.loss_function import reconstruct_8_corners_from_6d
from utils.iou_3d import compute_iou
from utils.preprocessing import compute_bbox_prior
from torch.utils.data import DataLoader


ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


if __name__ == "__main__":
    # python inference_onnx.py --config config/params_early_fusion.yaml --onnx checkpoints/model_early_fusion.onnx
    parser = argparse.ArgumentParser(description="Run ONNX inference on GPU for detection model")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to config yaml (e.g. config/params_early_fusion.yaml or params_late_fusion.yaml)")
    parser.add_argument('--onnx', type=str, required=True,
                        help="Path to ONNX model file")
    parser.add_argument('--fp16', action='store_true',
                        help="Set if ONNX model was exported in half precision")
    args = parser.parse_args()

    # Load configuration and create validation dataset.
    params = load_params(args.config)
    val_dataset = Bbox3DDataset(data_path=params.dataset.data_dir, params=params, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=detection_collate)

    # Create an ONNXRuntime session with GPU (CUDA) execution provider.
    session = onnxruntime.InferenceSession(args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name_rgb = session.get_inputs()[0].name
    input_name_pc = session.get_inputs()[1].name
    input_name_mask = session.get_inputs()[2].name
    output_name = session.get_outputs()[0].name

    inference_times = []
    all_ious = []
    print("Starting ONNX inference on GPU ...")
    for batch_idx, batch in enumerate(val_loader):
        # Retrieve inputs from the batch.
        rgb = batch["rgb"].numpy()  # shape: (1, 3, H, W)
        pc_list = batch["pc"][0]      # shape: (n_obj, n_sample, 3)
        mask_list = batch["mask"][0]  # shape: (n_obj, H, W)
        gt_bboxes = batch["bbox3d"][0].numpy()  # shape: (n_obj, 8, 3)

        rgb_np = rgb.astype(np.float16 if args.fp16 else np.float32)
        n_obj = pc_list.shape[0]
        pred_params_list = []
        start_time = time.time()
        # Process each object individually (the ONNX model expects a single object per run).
        for i in range(n_obj):
            pc_tensor = pc_list[i:i+1].numpy()    # shape: (1, n_sample, 3)
            mask_tensor = mask_list[i:i+1].numpy()  # shape: (1, H, W)
            if args.fp16:
                pc_np = pc_tensor.astype(np.float16)
                mask_np = mask_tensor.astype(np.float16)
            else:
                pc_np = pc_tensor.astype(np.float32)
                mask_np = mask_tensor.astype(np.float32)
            outputs = session.run([output_name],
                                  {input_name_rgb: rgb_np,
                                   input_name_pc: pc_np,
                                   input_name_mask: mask_np})
            # outputs[0] is assumed to be of shape (1, 12)
            pred_params_list.append(outputs[0][0])
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # For each predicted object, reconstruct the predicted 8-corner box.
        pred_corners_list = []
        for i, pred_params in enumerate(pred_params_list):
            center_prior, dims_prior, orient_6d_prior = compute_bbox_prior(pc_list[i].numpy(), device=torch.device('cpu'))
            center_pred = center_prior + torch.tensor(pred_params[0:3])
            dims_pred = dims_prior + torch.tensor(pred_params[3:6])
            orient_pred = orient_6d_prior + torch.tensor(pred_params[6:12])
            corners_pred = reconstruct_8_corners_from_6d(center_pred, dims_pred, orient_pred)
            pred_corners_list.append(corners_pred.numpy())
        # Compute IoU for each object.
        for i in range(n_obj):
            iou = compute_iou(gt_bboxes[i], pred_corners_list[i])
            all_ious.append(iou)

    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0
    mean_inference_time = np.mean(inference_times) if len(inference_times) > 0 else 0.0
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean inference time per sample: {mean_inference_time:.4f} seconds")
