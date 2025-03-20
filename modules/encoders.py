import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from box import ConfigBox
from torchvision.models.resnet import ResNet18_Weights

from .pointNet import PointNet


class ImageEncoder(nn.Module):
    def __init__(self, params: ConfigBox, d_model=256, load_pretrained=True):
        super().__init__()

        backbone_name = params.backbone

        if backbone_name == "resnet18":
            if load_pretrained:
                backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                backbone = models.resnet18(weights=None)
            # The feature-extractor part is everything except the final FC:
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        if params.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(16 * 22, d_model)

        print("Number of image encoder parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (torch.Tensor): (B, 3, H, W), e.g. (B, 3, 715,1003) after padding
        # Returns: torch.Tensor: (B, H'*W', d_model)
        # e.g. shape (B, 512, H', W')
        features = self.backbone(x)

        B, C, Hf, Wf = features.shape  # ([1, 512, 16, 22])
        features = features.view(B, C, -1)  # (B, 512, H'*W')
        features = self.proj(features)  # (B, 512, d_model)
        return features


class ImageMaskFusion(nn.Module):
    def __init__(self, params: ConfigBox, d_model=256, load_pretrained=True):
        super().__init__()

        # 1) Build or load the ResNet18 backbone
        if load_pretrained:
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=None)
        modules = list(backbone.children())

        self.layer0 = nn.Sequential(*modules[:4])  # conv1,bn1,relu,maxpool
        self.layer1 = modules[4]  # stride=4
        self.layer2 = modules[5]  # stride=8
        self.layer3 = modules[6]  # stride=16
        self.layer4 = modules[7]  # stride=32

        # Optionally freeze
        if params.freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

        # A projection for the pooled features from layer3 (256‐D) and layer4 (512‐D).
        #    We’ll combine them. Example: concat => (256 + 512) => d_model
        self.proj = nn.Linear(256 + 512, d_model)

        print("Number of image mask fusion encoder parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())))

    def encode(self, x: torch.Tensor):
        """
        Returns two feature maps: stage1 (stride=16), stage2 (stride=32).
        x: (B, 3, H, W)
        """
        # Pass through initial layers
        x = self.layer0(x)   # stride=4
        x = self.layer1(x)   # stride=4
        x = self.layer2(x)   # stride=8

        # Stage1: after layer3 => stride=16
        stage1 = self.layer3(x)  # (B, 256, H1, W1)

        # Stage2: after layer4 => stride=32
        stage2 = self.layer4(stage1)  # (B, 512, H2, W2)

        return stage1, stage2

    def mask_fusion_and_projection(self, feats_stage1, feats_stage2, mask: torch.Tensor) -> torch.Tensor:
        """
        Fuse the two‐stage feature maps with the object masks, then combine into d_model.

        Args:
            feats_stage1: (B, 256, H1, W1) from layer3
            feats_stage2: (B, 512, H2, W2) from layer4
            mask: (B, N, H, W)  (boolean or 0/1), same resolution as the original input image,
                  which is bigger than (H1,W1) or (H2,W2).

        Returns:
            fused_emb: (B, N, d_model), one embedding per object.
        """
        B, N, H, W = mask.shape

        # Downsample the mask to match each stage's resolution
        # stride=16 => size=(H1, W1)
        _, _, H1, W1 = feats_stage1.shape
        mask_s1 = F.interpolate(mask.float(), size=(H1, W1), mode='nearest')  # (B, N, H1, W1)
        # stride=32 => size=(H2, W2)
        _, _, H2, W2 = feats_stage2.shape
        mask_s2 = F.interpolate(mask.float(), size=(H2, W2), mode='nearest')  # (B, N, H2, W2)

        desc_list = []
        for i in range(N):
            # mask_s1[:, i] => (B, H1, W1)
            mask_i_s1 = mask_s1[:, i].unsqueeze(1)  # (B,1,H1,W1)
            masked_feat_s1 = feats_stage1 * mask_i_s1  # (B,256,H1,W1)
            # pool => shape (B,256)
            desc1 = masked_feat_s1.view(B, 256, -1).mean(dim=-1)

            mask_i_s2 = mask_s2[:, i].unsqueeze(1)  # (B,1,H2,W2)
            masked_feat_s2 = feats_stage2 * mask_i_s2  # (B,512,H2,W2)
            desc2 = masked_feat_s2.view(B, 512, -1).mean(dim=-1)

            # Combine them => shape (B, 256+512)
            combined = torch.cat([desc1, desc2], dim=-1)

            # Append for object i
            desc_list.append(combined)

        # Stack along object dimension => (B, N, 256+512)
        desc_all = torch.stack(desc_list, dim=1)

        # Final projection => (B, N, d_model)
        fused_emb = self.proj(desc_all)

        return fused_emb



class MaskEncoder(nn.Module):
    """
    Encodes N instance masks (each (H,W)) into a single embedding per mask.
    Input shape:  (B, N, H, W)
    Output shape: (B, N, d_model)
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.conv = nn.Sequential(
            # Add BatchNorm to help with binary input distribution
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B*n, 64, h//2, w//2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B*n, 128, h//4, w//4)
        )
        self.project = nn.Linear(128, d_model)
        print("Number of mask encoder parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())))

    def forward(self, mask_batch):
        """
        mask_batch: (B, N, H, W)
        We want (B, N, d_model).
        """
        B, N, H, W = mask_batch.shape

        mask_batch = mask_batch.view(B * N, 1, H, W).float()  # (B*N, 1, H, W)

        feats = self.conv(mask_batch) # shape

        # Global average pool
        feats = feats.view(B * N, 128, -1).mean(dim=2)  # (B*N, 128)
        feats = self.project(feats)  # (B*N, d_model)
        feats = feats.view(B, N, -1)  # (B, N, d_model)
        return feats


class PerObjectPointCloudEncoder(nn.Module):
    """
    Encodes a batch of per-object point clouds of shape (B, N, S, 3):
      B = batch size
      N = number of objects
      S = number of sampled points per object

    Produces (B, N, d_model).
    """
    def __init__(self, d_model=256, hidden_dim=512, dropout=0.1, encoder_type="mlp"):
        super().__init__()
        self.encoder_type = encoder_type
        # A small MLP (PointNet style) that processes each point, then we pool
        if encoder_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
        elif encoder_type == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(3, 64, 1), # 共享MLP
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        elif encoder_type == "pointnet":
            self.encoder = PointNet(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.final_proj = nn.Linear(hidden_dim, d_model)
        print("Number of point cloud encoder parameters: {:e}".format(
            sum(p.numel() for p in self.parameters())))

    def forward(self, pc_b_n_s_3: torch.Tensor) -> torch.Tensor:
        """
        pc_b_n_s_3: shape (B, N, S, 3)
        Return: (B, N, d_model)
        """
        B, N, S, _ = pc_b_n_s_3.shape
        if self.encoder_type == "mlp":
            pc_flat = pc_b_n_s_3.view(B*N*S, 3)  # (B*N*S, 3)
            feats = self.mlp(pc_flat)            # (B*N*S, hidden_dim)
            pooled = feats.view(B*N, S, -1).mean(dim=1)  # (B*N, hidden_dim)
        elif self.encoder_type == "cnn":
            pc_b_n_s_3 = pc_b_n_s_3.view(B*N, S, 3).transpose(1, 2)  # (B*N, 3, S)
            pooled = self.encoder(pc_b_n_s_3).squeeze(-1)  # (B*N, hidden_dim)
        elif self.encoder_type == "pointnet":
            pc_b_n_s_3 = pc_b_n_s_3.view(B*N, S, 3)  # (B*N, S, 3)
            pooled = self.encoder(pc_b_n_s_3)  # (B*N, 512)

        emb = self.final_proj(pooled)        # (B*N, d_model)
        # Reshape => (B, N, d_model)
        emb = emb.view(B, N, -1)
        return emb