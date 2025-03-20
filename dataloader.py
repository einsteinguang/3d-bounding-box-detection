import os
import sys
import torch
import numpy as np
from box import ConfigBox
from PIL import Image
from torch.utils.data import Dataset

from utils.transformation import get_transforms
from config.load_param import load_params
from modules.encoders import ImageEncoder
from modules.transformer_decoder import BBoxDetectionNetEarlyFusion


ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def load_single_sample(folder_path):
    sample = {}
    # Load RGB image
    rgb_path = os.path.join(folder_path, 'rgb.jpg')
    sample['rgb'] = np.array(Image.open(rgb_path))
    sample['bbox3d'] = np.load(os.path.join(folder_path, 'bbox3d.npy'))
    sample['mask'] = np.load(os.path.join(folder_path, 'mask.npy'))
    sample['pc'] = np.load(os.path.join(folder_path, 'pc.npy'))
    return sample


def detection_collate(batch):
    """
    Expects a list of dicts. Each dict has:
      'rgb':   (3, H, W)  # all images might have the same shape or not
      'pc':    (n_i, n_sample, 3)
      'mask':  (n_i, H, W)
      'bbox3d':(n_i, 8, 3)
    We want to produce a single dict for the batch.
    """
    # Collect them
    # If all rgb are same size, we can do a torch.stack for 'rgb'
    rgbs = [item['rgb'] for item in batch]
    batch_rgb = torch.stack(rgbs, dim=0)  # (B, 3, H, W)

    # But for pc, mask, bbox3d, they vary in n_i. We store each as a list
    pcs = [item['pc'] for item in batch]       # list of length B
    masks = [item['mask'] for item in batch]
    bbox3ds = [item['bbox3d'] for item in batch]

    return {
        'rgb': batch_rgb,
        'pc': pcs,
        'mask': masks,
        'bbox3d': bbox3ds
    }


class Bbox3DDataset(Dataset):
    def __init__(self, data_path, params: ConfigBox, train=False):
        self.data_path = data_path
        self.train = train
        self.params = params
        self.folders = [f.path for f in os.scandir(data_path) if f.is_dir()]
        # folders ends with "data_1" - "data_200", now sort folders
        self.folders.sort(key=lambda x: int(x.split('_')[-1]))
        # shuffle the folders
        # np.random.shuffle(self.folders)
        if train:
            self.transforms = get_transforms(is_train=True, params=params.dataset.transforms)
            self.folders = self.folders[:int(len(self.folders) * params.dataset.train_ratio)]
        else:
            self.transforms = get_transforms(is_train=False, params=params.dataset.transforms)
            self.folders = self.folders[int(len(self.folders) * params.dataset.train_ratio):]

    def check_bounds(self):
        # No random scale shift, point cloud bounds:
        # X: [-0.912, 0.720]
        # Y: [-0.466, 0.611]
        # Z: [-1.000, 2.000]
        # bounding box bounds:
        # X: [-0.350, 0.324]
        # Y: [-0.284, 0.276]
        # Z: [-0.479, 0.428]

        # Normal 0.3 scale + 0.5 shift, point cloud bounds:
        # X: [-1.596, 1.733]
        # Y: [-1.417, 1.437]
        # Z: [-2.512, 3.182]
        # bounding box bounds:
        # X: [-1.727, 1.677]
        # Y: [-1.926, 1.318]
        # Z: [-1.660, 1.580]

        # Normal 0.5 scale + 1.0 shift, point cloud bounds:
        # X: [-2.088, 1.546]
        # Y: [-1.197, 1.565]
        # Z: [-2.143, 3.424]
        # bounding box bounds:
        # X: [-1.321, 1.318]
        # Y: [-1.192, 1.248]
        # Z: [-1.288, 1.379]
        print(f"Checking bounds for {params.dataset.transforms.style}")
        min_vals = np.array([float('inf')] * 3)
        max_vals = np.array([float('-inf')] * 3)
        dataset = self
        for i in range(len(dataset)):
            sample = dataset[i]
            # pc is tensor (n, n_sample, 3)
            pc = sample['pc']
            pc_numpy = pc.numpy()  # Convert tensor to numpy
            # Reshape to (n*n_sample, 3) to get all points
            pc_reshaped = pc_numpy.reshape(-1, 3)

            min_vals = np.minimum(min_vals, pc_reshaped.min(axis=0))
            max_vals = np.maximum(max_vals, pc_reshaped.max(axis=0))

        print(f"Transformed point cloud bounds:")
        print(f"X: [{min_vals[0]:.3f}, {max_vals[0]:.3f}]")
        print(f"Y: [{min_vals[1]:.3f}, {max_vals[1]:.3f}]")
        print(f"Z: [{min_vals[2]:.3f}, {max_vals[2]:.3f}]")

        # Check bbox3d bounds
        min_vals_bbox = np.array([float('inf')] * 3)
        max_vals_bbox = np.array([float('-inf')] * 3)

        for i in range(len(dataset)):
            sample = dataset[i]
            bbox3d = sample['bbox3d']
            bbox3d_numpy = bbox3d.numpy()  # Convert tensor to numpy
            # bbox3d is already in shape (n, 8, 3), reshape to (n*8, 3)
            bbox_reshaped = bbox3d_numpy.reshape(-1, 3)

            min_vals_bbox = np.minimum(min_vals_bbox, bbox_reshaped.min(axis=0))
            max_vals_bbox = np.maximum(max_vals_bbox, bbox_reshaped.max(axis=0))

        print(f"\nTransformed bounding box bounds:")
        print(f"X: [{min_vals_bbox[0]:.3f}, {max_vals_bbox[0]:.3f}]")
        print(f"Y: [{min_vals_bbox[1]:.3f}, {max_vals_bbox[1]:.3f}]")
        print(f"Z: [{min_vals_bbox[2]:.3f}, {max_vals_bbox[2]:.3f}]")

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = self.folders[idx]
        # check if folder_path exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Data {folder_path} does not exist")
        sample = load_single_sample(folder_path)
        sample = self.transforms(sample)
        # rgb: tensor (C, h, w) in float [0, 1]
        # mask: tensor (n, H, W) bool
        # pc: tensor (n, n_sample, 3)
        # bbox3d': tensor (n, 8, 3)
        return sample


def check_transformed_pc_bounds(params):
    data_path = "dl_challenge"
    dataset = Bbox3DDataset(data_path, params, train=True)
    dataset.check_bounds()


def test_image_encoder(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "dl_challenge"
    dataset = Bbox3DDataset(data_path, params, train=True)

    # dataloader test
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=5, collate_fn=detection_collate, shuffle=True)

    image_encoder = ImageEncoder(params.model.image_encoder, params.model.d_model, True).to(device)
    for batch in dataloader:
        rgb = batch["rgb"].to(device)
        out = image_encoder(rgb)
        print(out.shape)
        break


def test_dataset(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "dl_challenge"
    dataset = Bbox3DDataset(data_path, params, train=True)

    # dataloader test
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=5, collate_fn=detection_collate, shuffle=True)

    detection_net = BBoxDetectionNetEarlyFusion(params.model, True).to(device)
    for batch in dataloader:
        rgb = batch["rgb"].to(device)
        pc = [x.to(device) for x in batch["pc"]]
        mask = [x.to(device) for x in batch["mask"]]
        bbox3d = [x.to(device) for x in batch["bbox3d"]]
        out = detection_net(rgb, pc, mask)
        print(f"Output shape: {out[0].shape}")
        break


if __name__ == "__main__":
    params_path = "config/params_9.yaml"
    params = load_params(params_path)
    test_dataset(params)