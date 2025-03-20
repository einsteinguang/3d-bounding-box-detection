import torch
import argparse
from config.load_param import load_params
from modules.transformer_decoder import BBoxDetectionNetEarlyFusion, BBoxDetectionNetLateFusion

# A wrapper to make the model’s inputs ONNX-exportable.
# Your original forward accepts lists for the point cloud and mask,
# so we wrap it to convert single tensor inputs into one-element lists.
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, rgb, pc, mask):
        # pc and mask are assumed to be single-tensor inputs (for one object)
        # We convert them to lists and return the first (and only) detection.
        out = self.model(rgb, [pc], [mask])
        # out is a list with one element (for one image); return that element.
        return out[0]

if __name__ == "__main__":
    # python convert_to_onnx.py --config config/params_late_fusion.yaml --checkpoint checkpoints/model_late_fusion.pth --output checkpoints/model_late_fusion.onnx
    parser = argparse.ArgumentParser(description="Convert PyTorch detection model to ONNX")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to config yaml (e.g. config/params_early_fusion.yaml or params_late_fusion.yaml)")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint (.pth file)")
    parser.add_argument('--output', type=str, required=True,
                        help="Output ONNX filename")
    parser.add_argument('--fp16', action='store_true',
                        help="Convert model and inputs to half precision")
    args = parser.parse_args()

    # Load config and create model.
    params = load_params(args.config)
    if hasattr(params.model, 'fusion_style') and params.model.fusion_style == "early_fusion":
        model = BBoxDetectionNetEarlyFusion(params.model, load_pretrained=False)
    else:
        model = BBoxDetectionNetLateFusion(params.model, load_pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    if args.fp16:
        model.half()

    # Wrap model to produce a forward signature that takes only tensors.
    wrapper = ModelWrapper(model)

    # Create dummy inputs. We use batch size=1.
    # Use the padded image dimensions from the config.
    H = int(params.dataset.transforms.padding_size_h * params.dataset.transforms.img_down_scale)
    W = int(params.dataset.transforms.padding_size_w * params.dataset.transforms.img_down_scale)
    n_points = params.dataset.transforms.n_sample_pc
    dummy_rgb = torch.randn(1, 3, H, W)
    dummy_pc = torch.randn(1, n_points, 3)  # one object with 1024 points
    dummy_mask = torch.randn(1, H, W)     # one mask

    if args.fp16:
        dummy_rgb = dummy_rgb.half()
        dummy_pc = dummy_pc.half()
        dummy_mask = dummy_mask.half()

    # Export the model.
    torch.onnx.export(
        wrapper,
        (dummy_rgb, dummy_pc, dummy_mask),
        args.output,
        input_names=["rgb", "pc", "mask"],
        output_names=["bbox_pred"],
        dynamic_axes={
            "rgb": {0: "batch", 2: "height", 3: "width"},
            "pc": {0: "n_obj", 1: "n_sample"},
            "mask": {0: "n_obj", 1: "height", 2: "width"},
            "bbox_pred": {0: "n_obj"}
        },
        opset_version=11
    )
    print(f"Model exported to {args.output}")
