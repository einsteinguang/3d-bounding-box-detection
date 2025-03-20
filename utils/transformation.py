import torch
import numpy as np
import random
from box import ConfigBox
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


def get_transforms(is_train, params: ConfigBox):
    """
    Returns a transforms pipeline that may include
    color jitter, random flips, random crop, gaussian noise, etc.
    """
    # We’ll build a list of transforms step by step
    transform_list = []

    # 1) Convert to PIL
    transform_list.append(ScaleAndShift3D(
        scale=params.scale,
        sx=params.sx,
        sy=params.sy,
        sz=params.sz
    ))
    if is_train:
        transform_list.append(RandomScaleAndShift3D(
            scale_var=params.scale_var,
            shift_var_x=params.shift_var_x,
            shift_var_y=params.shift_var_y,
            shift_var_z=params.shift_var_z,
            style=params.style
        ))
    transform_list.append(ConvertToPILMulti())

    if is_train:
        if params.use_color_jitter:
            transform_list.append(
                ColorJitterMulti(
                    brightness=params.color_jitter.brightness,
                    contrast=params.color_jitter.contrast,
                    saturation=params.color_jitter.saturation,
                    hue=params.color_jitter.hue
                )
            )

    transform_list.append(
        PadMulti(target_size=(params.padding_size_h, params.padding_size_w),
                 fill=0,
                 padding_mode=params.padding_mode))
    transform_list.append(SampleFixedNumberOfPoints(n_sample=params.n_sample_pc))
    transform_list.append(DownscaleImages(img_down_scale=params.img_down_scale))
    transform_list.append(ConvertToTensorMulti())

    if is_train:
        if params.use_gaussian_noise:
            transform_list.append(GaussianNoiseImage(std=params.gaussian_noise_std))

        if params.use_pc_noise:
            transform_list.append(RandomPointCloudNoise(std=params.pc_noise_std))
    return T.Compose(transform_list)

# --------------------------------------------------------------------
# Below are custom classes that handle the dictionary-based sample:
#   sample = {
#       'rgb':   (h, w, 3),   uint8
#       'mask':  (n, h, w),   bool
#       'pc':    (3, h, w),   float64
#       'bbox3d':(n, 8, 3),   float32
#   }
# We want to apply the same spatial transforms to 'rgb', 'mask' (n channels),
# and 'pc' (3 channels) but keep 'bbox3d' untouched.
# --------------------------------------------------------------------

class ScaleAndShift3D:
    """
    Applies a (scale_x, scale_y, scale_z) and (shift_x, shift_y, shift_z)
    The scaling is done first, then the shift is added.
    """
    def __init__(self,
                 scale=0.65,
                 sx=0., sy=0., sz=-1.5):
        self.scale = scale
        self.tx, self.ty, self.tz = sx, sy, sz

    def __call__(self, sample):
        pc = sample["pc"]
        pc[0, :, :] *= self.scale
        pc[1, :, :] *= self.scale
        pc[2, :, :] *= self.scale
        pc[0, :, :] += self.tx
        pc[1, :, :] += self.ty
        pc[2, :, :] += self.tz
        sample["pc"] = pc

        bbox_3d = sample["bbox3d"]
        bbox_3d[:, :, 0] *= self.scale
        bbox_3d[:, :, 1] *= self.scale
        bbox_3d[:, :, 2] *= self.scale
        bbox_3d[:, :, 0] += self.tx
        bbox_3d[:, :, 1] += self.ty
        bbox_3d[:, :, 2] += self.tz
        sample["bbox3d"] = bbox_3d
        return sample


class RandomScaleAndShift3D:
    def __init__(self,
                 scale_var=0.1,
                 shift_var_x=0.1,
                 shift_var_y=0.1,
                 shift_var_z=0.1,
                 style="normal"):
        self.scale_var = scale_var
        self.shift_var_x = shift_var_x
        self.shift_var_y = shift_var_y
        self.shift_var_z = shift_var_z
        self.style = style

    def __call__(self, sample):
        if self.style == "normal":
            scale = 1.0 + np.random.normal(0, self.scale_var)
            tx = np.random.normal(0, self.shift_var_x)
            ty = np.random.normal(0, self.shift_var_y)
            tz = np.random.normal(0, self.shift_var_z)
        elif self.style == "uniform":
            scale = 1.0 + np.random.uniform(-self.scale_var, self.scale_var)
            tx = np.random.uniform(-self.shift_var_x, self.shift_var_x)
            ty = np.random.uniform(-self.shift_var_y, self.shift_var_y)
            tz = np.random.uniform(-self.shift_var_z, self.shift_var_z)
        else:
            raise ValueError(f"Unknown style for RandomScaleAndShift3D: {self.style}")

        # Apply to point cloud
        pc = sample["pc"]
        pc[0, :, :] *= scale
        pc[1, :, :] *= scale
        pc[2, :, :] *= scale
        pc[0, :, :] += tx
        pc[1, :, :] += ty
        pc[2, :, :] += tz
        sample["pc"] = pc

        # Apply to bounding boxes
        bbox_3d = sample["bbox3d"]
        bbox_3d[:, :, 0] *= scale
        bbox_3d[:, :, 1] *= scale
        bbox_3d[:, :, 2] *= scale
        bbox_3d[:, :, 0] += tx
        bbox_3d[:, :, 1] += ty
        bbox_3d[:, :, 2] += tz
        sample["bbox3d"] = bbox_3d
        return sample


class ConvertToPILMulti:
    """
    1) Convert rgb (H,W,3) [uint8] to a PIL Image.
    2) Convert each mask channel (H,W) to a separate PIL Image (or stay as a list).
    3) Convert the point cloud (3,H,W) to a PIL Image so we can apply 2D flips/crops if desired.
       (We're effectively treating it like a 3-channel image for transforms.)
    """
    def __call__(self, sample):
        rgb = sample['rgb']        # shape (h, w, 3) dtype=uint8
        mask = sample['mask']      # shape (n, h, w) dtype=bool
        pc = sample['pc']          # shape (3, h, w) dtype=float64
        # bbox3d remains as is
        bbox3d = sample['bbox3d']

        # Convert rgb to PIL
        # PIL expects (H, W, 3) in uint8
        pil_rgb = Image.fromarray(rgb)

        # Convert each mask channel to its own PIL (8-bit or 1-bit).
        # Easiest is to cast bool->uint8, then to PIL. We store them in a list.
        pil_masks = []
        for i in range(mask.shape[0]):
            # shape: (h, w), bool -> uint8
            mask_i = (mask[i] * 255).astype(np.uint8)
            pil_masks.append(Image.fromarray(mask_i, mode='L'))

        # Convert pc to a “3-channel image”, float64 -> float32
        pc_np = pc.transpose(1, 2, 0).astype(np.float32)  # (h, w, 3)
        # Store as 3 single-channel float images in a list:
        pil_pcs = []
        for c in range(pc_np.shape[2]):
            pc_chan = pc_np[..., c]  # shape (h, w)
            # Convert each channel to its own float image
            pil_pcs.append(Image.fromarray(pc_chan, mode='F'))

        sample['rgb'] = pil_rgb
        sample['mask'] = pil_masks  # list of PIL
        sample['pc'] = pil_pcs      # list of PIL (3 channels)
        sample['bbox3d'] = bbox3d   # unchanged
        return sample


class ConvertToTensorMulti:
    """
    Converts:
      - 'rgb' PIL -> Torch tensor [3, H, W] in float
      - 'mask' list of PIL -> single bool tensor (n, H, W)
      - 'pc' numpy array (n_object, n_sample, 3) -> just use torch.from_numpy
      - 'bbox3d' is converted to torch if it’s numpy.
    """
    def __call__(self, sample):
        # Convert rgb: PIL -> Tensor (C,H,W) in float [0,1]
        sample['rgb'] = F.to_tensor(sample['rgb'])

        # Convert mask list -> (n, H, W) bool
        mask_tensors = []
        for m in sample['mask']:
            mt = F.to_tensor(m)  # shape (1,H,W) float
            mt = mt.squeeze(0).bool()
            mask_tensors.append(mt)
        if mask_tensors:
            sample['mask'] = torch.stack(mask_tensors, dim=0)  # (n,H,W)
        else:
            sample['mask'] = torch.zeros(
                (0,) + sample['rgb'].shape[1:], dtype=torch.bool
            )

        # Convert pc: np array of shape (n, n_sample, 3)
        pc_np = sample['pc']
        sample['pc'] = torch.from_numpy(pc_np).float()  # (n, n_sample, 3)

        # bbox3d -> torch if still numpy
        if isinstance(sample['bbox3d'], np.ndarray):
            sample['bbox3d'] = torch.from_numpy(sample['bbox3d'])

        return sample


class ColorJitterMulti:
    """
    Applies Color Jitter to the 'rgb' image only, using torchvision's built-in transform.
    Leaves 'mask', 'pc', and 'bbox3d' unchanged.
    This must operate on PIL images (so call this after ConvertToPILMulti).
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, sample):
        # sample['rgb'] is a PIL image after ConvertToPILMulti
        sample['rgb'] = self.jitter(sample['rgb'])
        return sample


class GaussianNoiseImage:
    """
    Adds random Gaussian noise to the 'rgb' tensor (shape: [3, H, W]).
    Leaves 'mask', 'pc', 'bbox3d' unchanged.
    Must be called *after* ConvertToTensorMulti.
    """

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, sample):
        rgb = sample['rgb']
        # Ensure the image is float [0,1], shape [3, H, W]
        noise = torch.randn_like(rgb) * self.std
        sample['rgb'] = rgb + noise
        # clamp to [0,1] in case the noise pushes values outside
        sample['rgb'] = sample['rgb'].clamp(0.0, 1.0)
        return sample


class RandomPointCloudNoise:
    """
    Adds random Gaussian noise to each point in the point cloud tensor: shape [3, H, W].
    Must be called after ConvertToTensorMulti so `pc` is a torch.Tensor.
    """

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, sample):
        pc = sample['pc']  # shape [n_object, 3]
        noise = torch.randn_like(pc) * self.std
        sample['pc'] = pc + noise
        return sample


class PadMulti:
    def __init__(self, target_size=(715, 1003), fill=0, padding_mode='constant'):
        """
        Args:
            target_size (tuple): (target_h, target_w) to pad to.
            fill (int or tuple): Pixel fill value for constant fill.
                                 For PC channels, 'fill' just means float 0 if using mode='constant'.
            padding_mode (str): 'constant', 'edge', 'reflect', or 'symmetric'
        """
        self.target_h = target_size[0]
        self.target_w = target_size[1]
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        """
        sample keys:
          - 'rgb': PIL Image (H,W)
          - 'mask': list of PIL Images (H,W)
          - 'pc': list of PIL Images (H,W) in mode='F'
        """
        rgb = sample['rgb']
        masks = sample['mask']
        pcs = sample['pc']

        # For each image, we find how much to pad on bottom/right to reach self.target_h, self.target_w
        w, h = rgb.size  # PIL size => (width, height)
        pad_h = max(0, self.target_h - h)
        pad_w = max(0, self.target_w - w)

        # Padding in torchvision is (left, top, right, bottom)
        padding = (0, 0, pad_w, pad_h)  # zero on top-left, fill on right-bottom

        # 1) Pad the RGB
        if pad_w > 0 or pad_h > 0:
            rgb = F.pad(rgb, padding, fill=self.fill, padding_mode=self.padding_mode)

        # 2) Pad each mask channel
        padded_masks = []
        for m in masks:
            padded_masks.append(F.pad(m, padding, fill=self.fill, padding_mode=self.padding_mode))

        # 3) Pad each pc channel
        padded_pcs = []
        for pc_chan in pcs:
            padded_pcs.append(F.pad(pc_chan, padding, fill=self.fill, padding_mode=self.padding_mode))

        sample['rgb'] = rgb
        sample['mask'] = padded_masks
        sample['pc'] = padded_pcs

        return sample


class DownscaleImages:
    """
    Downscales 'rgb', 'mask', and 'pc' images by the given factor.
    Must be called when the data is still in PIL Image format.
    """
    def __init__(self, img_down_scale=0.5):
        self.scale = img_down_scale

    def __call__(self, sample):
        # Get original size
        rgb = sample['rgb']
        w, h = rgb.size
        # Calculate new size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        # Resize RGB
        sample['rgb'] = rgb.resize((new_w, new_h), Image.BILINEAR)

        # Resize each mask channel
        resized_masks = []
        for m in sample['mask']:
            resized_masks.append(m.resize((new_w, new_h), Image.NEAREST))
        sample['mask'] = resized_masks
        return sample


class SampleFixedNumberOfPoints:
    """
    For each object mask, gather its 3D points from the depth camera (pc),
    then sample a fixed number n_sample of points.
    If an object has fewer than n_sample points, we randomly duplicate some.
    If an object has more, we randomly downsample.

    The result is stored in sample["pc"] with shape (n_object, n_sample, 3).
    """
    def __init__(self, n_sample=1024):
        self.n_sample = n_sample

    def __call__(self, sample):
        """
        sample["pc"]:   (3, H, W) float
        sample["mask"]: (n_object, H, W) bool
        We produce sample["pc"] = (n_object, n_sample, 3).
        """
        pc_3hw = np.array(sample["pc"] ) # shape (3, H, W), float
        masks = np.array(sample["mask"])  # shape (n_object, H, W), bool

        # Reshape pc to (H*W, 3) for easy indexing
        H, W = pc_3hw.shape[1], pc_3hw.shape[2]
        pc_hw3 = pc_3hw.transpose(1,2,0).reshape(-1, 3)  # (H*W, 3)

        # For each object:
        pc_per_object = []
        for obj_idx in range(masks.shape[0]):
            mask_hw = masks[obj_idx]  # (H, W)
            mask_flat = mask_hw.reshape(-1).astype(bool)  # (H*W,)
            # gather object points
            obj_points = pc_hw3[mask_flat]  # shape (~num_pts, 3)
            num_pts = obj_points.shape[0]

            if num_pts == 0:
                # print sum of mask as int
                print(f"Sum of mask: {mask_flat.sum().astype(int)}")
                print(f"Warning: object {obj_idx} has no points in the point cloud.")
                continue

            if num_pts >= self.n_sample:
                # Downsample
                idxs = np.random.choice(num_pts, self.n_sample, replace=False)
            else:
                # Upsample with replacement
                idxs = np.random.choice(num_pts, self.n_sample, replace=True)
            sampled_points = obj_points[idxs]
            pc_per_object.append(sampled_points.astype(np.float32))

        # Stack => (n_object, n_sample, 3)
        pc_per_object = np.stack(pc_per_object, axis=0)
        sample["pc"] = pc_per_object  # shape (n_object, n_sample, 3)
        return sample
