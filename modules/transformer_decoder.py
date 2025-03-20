import torch
import torch.nn as nn
from box import ConfigBox

from .encoders import *

class CombinedQueryBuilder(nn.Module):
    """
    Combines point cloud encoding and mask encoding into a single query embedding
    for each object.
    """
    def __init__(self, pc_dim=256, mask_dim=256, d_model=256):
        super().__init__()
        self.fc_combine = nn.Sequential(
            nn.Linear(pc_dim + mask_dim, d_model),
        )

    def forward(self, pc_emb, mask_emb):
        """
        pc_emb:   (B, N, pc_dim)
        mask_emb: (B, N, mask_dim)
        Returns: (B, N, d_model)
        """
        combined = torch.cat([pc_emb, mask_emb], dim=-1)
        out = self.fc_combine(combined)
        return out


class BBoxDetectionNetLateFusion(nn.Module):
    def __init__(self, params: ConfigBox, load_pretrained):
        super().__init__()
        d_model = params.d_model
        self.img_encoder = ImageEncoder(params.image_encoder, d_model, load_pretrained)  # (B, L, d_model)

        self.pc_encoder = PerObjectPointCloudEncoder(
            d_model=params.pc_output_dim,
            hidden_dim=params.pc_hidden_dim,
            dropout=params.pc_encoder_dropout,
            encoder_type=params.pc_encoder_type)  # returns (B, N, 256)
        self.mask_encoder = MaskEncoder(d_model=params.mask_output_dim)                 # returns (B, N, 256)
        self.query_builder = CombinedQueryBuilder(
            pc_dim=params.pc_output_dim, mask_dim=params.mask_output_dim, d_model=d_model)

        # e.g. a standard Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=params.n_heads, dropout=params.dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=params.n_decoder_layers)

        # bounding box head
        self.bbox_head = nn.Sequential(
            nn.Linear(params.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 12),  # center (3) + dims (3) + orient_6d (6)
        )
        print("Number of total parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def forward(self, rgb, pcs, masks):
        # rgb: (B, 3, H, W)
        # pc:  list of (N, S, 3)
        # mask: list of (N, H, W)

        # Image -> (B, L, d_model)
        img_feats = self.img_encoder(rgb)

        B = img_feats.size(0)  # batch size

        out_boxes = []

        for i in range(B):
            pc = pcs[i].unsqueeze(0)
            mask = masks[i].unsqueeze(0)

            # For each object, gather the 3D points => (1, N, pc_output_dim)
            pc_emb = self.pc_encoder(pc)

            # Encode mask => (1, N, mask_output_dim)
            mask_emb = self.mask_encoder(mask)

            # Combine into final queries => (1, N, d_model)
            queries = self.query_builder(pc_emb, mask_emb)

            # decode
            # print(queries.shape, img_feats[i].unsqueeze(0).shape)
            decoded = self.decoder(tgt=queries, memory=img_feats[i].unsqueeze(0))  # (1, N, d_model)

            # bounding box regression
            out_box = self.bbox_head(decoded).squeeze(0)  # (N, 12)

            out_boxes.append(out_box)

        return out_boxes


class BBoxDetectionNetEarlyFusion(nn.Module):
    def __init__(self, params: ConfigBox, load_pretrained):
        super().__init__()
        d_model = params.d_model
        self.fusion_style_second_stage = params.fusion_style_second_stage
        self.img_encoder = ImageMaskFusion(params.image_encoder, d_model, load_pretrained)  # (B, L, d_model)

        self.pc_encoder = PerObjectPointCloudEncoder(
            d_model=params.pc_output_dim,
            hidden_dim=params.pc_hidden_dim,
            dropout=params.pc_encoder_dropout,
            encoder_type=params.pc_encoder_type)  # returns (B, N, 256)

        # e.g. a standard Transformer decoder
        if self.fusion_style_second_stage == "transformer":
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=params.n_heads, dropout=params.dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=params.n_decoder_layers)

        # bounding box head
        dim_before_head = d_model if self.fusion_style_second_stage == "transformer" else 2*d_model
        self.bbox_head = nn.Sequential(
            nn.Linear(dim_before_head, 128),
            nn.ReLU(),
            nn.Linear(128, 12),  # center (3) + dims (3) + orient_6d (6)
        )
        print("Number of total parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def forward(self, rgb, pcs, masks):
        # rgb: (B, 3, H, W)
        # pc:  list of (N, S, 3)
        # mask: list of (N, H, W)

        # Image -> (B, L, d_model)
        img_feats1, img_feats2 = self.img_encoder.encode(rgb)

        B = rgb.size(0)  # batch size

        out_boxes = []

        for i in range(B):
            pc = pcs[i].unsqueeze(0)
            mask = masks[i].unsqueeze(0)

            # Fuse the image and mask features
            fused_embedding = self.img_encoder.mask_fusion_and_projection(
                img_feats1[i].unsqueeze(0), img_feats2[i].unsqueeze(0), mask)  # shape (1, N, d_model)

            # For each object, gather the 3D points => (1, N, d_model)
            pc_emb = self.pc_encoder(pc)

            if self.fusion_style_second_stage == "transformer":
                decoded = self.decoder(tgt=pc_emb, memory=fused_embedding)  # (1, N, d_model)
            elif self.fusion_style_second_stage == "concat":
                decoded = torch.cat([pc_emb, fused_embedding], dim=-1)  # (1, N, 2*d_model)

            # bounding box regression
            out_box = self.bbox_head(decoded).squeeze(0)  # (N, 12)

            out_boxes.append(out_box)

        return out_boxes
