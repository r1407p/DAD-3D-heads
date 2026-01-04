from typing import Dict, Any

import torch
import torch.nn as nn
from model_training.data.config import (
    OUTPUT_LANDMARKS_HEATMAP,
    OUTPUT_3DMM_PARAMS,
    OUTPUT_2D_LANDMARKS,
    OUTPUT_DEPTH,
    OUTPUT_REGION,
    OUTPUT_3D_VERTICES,
    OUTPUT_2D_VERTICES,
    OUTPUT_RESIDUAL_DEFORMATION,
    OUTPUT_3D_VERTICES_REFINED,
    OUTPUT_2D_VERTICES_REFINED,
)
from model_training.model.encoders import get_encoder
from model_training.model.bifpn import BiFPN
from model_training.model.layers import IdentityLayer
from model_training.head_mesh import HeadMesh
from torch.nn import functional as F

__all__ = ["FlameRegression"]


class FlameHead(IdentityLayer):
    def __init__(self, model_config, network_metadata):
        super().__init__(model_config=model_config, network_metadata=network_metadata)
        self.heatmap = nn.Conv2d(
            model_config["num_filters"], out_channels=model_config["num_classes"], kernel_size=3, padding=1
        )
        self.heatmap.bias.data.fill_(0.0)

    def forward(self, decoder_output):
        x = decoder_output[0]
        heatmap = self.heatmap(x)
        return heatmap


class FlameDepth(IdentityLayer):
    def __init__(self, model_config, network_metadata):
        super().__init__(model_config=model_config, network_metadata=network_metadata)
        self.depth = nn.Conv2d(
            model_config["num_filters"], out_channels=model_config["num_classes"], kernel_size=3, padding=1
        )
        self.depth.bias.data.fill_(0.0)

    def forward(self, decoder_output):
        x = decoder_output[0]
        depth = self.depth(x)
        return depth


class FlameRegion(IdentityLayer):
    def __init__(self, model_config, network_metadata):
        super().__init__(model_config=model_config, network_metadata=network_metadata)
        self.region = nn.Conv2d(
            model_config["num_filters"], out_channels=model_config["num_classes"], kernel_size=3, padding=1
        )
        self.region.bias.data.fill_(0.0)

    def forward(self, decoder_output):
        x = decoder_output[0]
        region = self.region(x)
        return region


class FusionLayer(nn.Module):
    def __init__(self, num_filters, num_heatmaps, output_filters, num_depth, num_region):
        super().__init__()
        self.conv1x1 = nn.Conv2d(num_filters + num_heatmaps + num_depth + num_region + output_filters, output_filters, kernel_size=1)

    def forward(self, x, heatmap, depth, region, bifpn_map):
        _, _, h, w = x.size()
        original_h = h if isinstance(h, int) else h.item()
        original_w = w if isinstance(w, int) else w.item()
        heatmap = nn.functional.interpolate(
            heatmap, size=(original_h, original_w), mode="bilinear", align_corners=True
        ).sigmoid()
        depth = nn.functional.interpolate(
            depth, size=(original_h, original_w), mode="bilinear", align_corners=True
        ).sigmoid()

        region = nn.functional.interpolate(
            region, size=(original_h, original_w), mode="bilinear", align_corners=True
        ).sigmoid()

        fmap = torch.cat([x, heatmap, depth, region, bifpn_map], dim=1)
        fmap = self.conv1x1(fmap)
        return fmap * x


class ClassificationHead(nn.Module):
    def __init__(self, num_filters, num_classes, dropout=0.3, linear_size=512):
        super().__init__()

        self.logit_image = nn.Sequential(
            nn.Linear(num_filters, linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(linear_size, num_classes),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        f = F.adaptive_avg_pool2d(x, output_size=1)
        return self.logit_image(f.view(batch_size, -1))


class ResidualDeformationHead(nn.Module):
    def __init__(self, in_dim, num_vertices, hidden_dim=256, scale=0.01):
        super().__init__()
        self.num_vertices = num_vertices
        self.scale = scale

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_vertices * 3),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat):
        """
        feat: (B, C)
        return: (B, V, 3)
        """
        B = feat.shape[0]
        delta = self.mlp(feat).view(B, self.num_vertices, 3)
        return delta * self.scale


class FlameRegression(nn.Module):
    def __init__(self, model_config: Dict[str, Any], consts_config: Dict[str, Any], num_classes: int = 68):
        super().__init__()
        self.encoder = get_encoder(model_config["backbone"], model_config.get("pretrained", False))
        self.bifpn = BiFPN(
            [
                self.encoder.encoder_channels["layer3"],
                self.encoder.encoder_channels["layer2"],
                self.encoder.encoder_channels["layer1"],
            ],
            model_config["num_filters"],
        )
        self.head = FlameHead(
            {"num_filters": model_config["num_filters"], "num_classes": model_config["num_classes"]}, {}
        )
        self.depth = FlameDepth(
            {"num_filters": model_config["num_filters"], "num_classes": 1}, {}
        )
        self.region = FlameRegion(
            {"num_filters": model_config["num_filters"], "num_classes": 1}, {}
        )
        self.max_layer = 4
        self.limit_value = model_config["limit_value"]
        self.fusion_layer = FusionLayer(
            model_config["num_filters"], model_config["num_classes"], self.encoder.encoder_channels["layer1"], 1, 1
        )

        self.shape = ClassificationHead(self.encoder.encoder_channels["layer0"], 403)
        self.pose = ClassificationHead(self.encoder.encoder_channels["layer0"], 10)
        self.landmarks = ClassificationHead(self.encoder.encoder_channels["layer0"], num_classes * 2)
        
        # HeadMesh for computing 3D vertices and 2D reprojected vertices
        self._img_size = model_config["img_size"]
        self.head_mesh = HeadMesh(
            flame_config=consts_config,
            batch_size=1,  # Will work with any batch size
            image_size=self._img_size
        )
        # freeze the head mesh
        for param in self.head_mesh.parameters():
            param.requires_grad = False

        num_vertices = 5023  # FLAME vertex count

        pre_residual_head_in_dim = model_config["num_filters"] + model_config["num_classes"] + 1 + 1 + self.encoder.encoder_channels["layer1"]
        self.pre_residual_head = nn.Conv2d(pre_residual_head_in_dim, 256, kernel_size=1)

        self.residual_head = ResidualDeformationHead(
            in_dim=2048,
            num_vertices=num_vertices,
            hidden_dim=256,
            scale=0.01
        )


    def forward(self, x):
        encoder_output = []
        for stage in self.encoder.stages[: self.max_layer]:
            x = stage(x)
            encoder_output.append(x)
        decoder_output = self.bifpn(encoder_output[1:])
        heatmap = self.head(decoder_output)
        depth = self.depth(decoder_output)
        region = self.region(decoder_output)
        fmap = self.fusion_layer(x, heatmap, depth, region, decoder_output[2])
        fmap = self.encoder.stages[-1](fmap)
        shape = self.shape(fmap).tanh() * self.limit_value
        pose = self.pose(fmap)
        landmarks = self.landmarks(fmap)
        B, N = landmarks.size()
        landmarks = F.relu(landmarks.reshape((B, N // 2, 2)), inplace=True)

        params_3dmm = torch.cat([shape, pose], dim=1)
        
        # Compute 3D vertices and 2D reprojected vertices from 3DMM params
        vertices_3d = self.head_mesh.vertices_3d(params_3dmm=params_3dmm, zero_rotation=True)
        vertices_2d = self.head_mesh.reprojected_vertices(params_3dmm=params_3dmm, to_2d=True)


        res_feat = F.adaptive_avg_pool2d(fmap, 1).view(B, -1)

        # maps = torch.cat([x, heatmap, depth, region, decoder_output[2]], dim=1)
        # map = self.pre_residual_head(maps)
        # res_feat = F.adaptive_avg_pool2d(map, 1).view(B, -1)
        residual_deformation = self.residual_head(res_feat)
        vertices_3d_refined = vertices_3d + residual_deformation
        vertices_2d_refined = self.head_mesh.reprojected_vertices_from_vertices_3d(vertices_3d_refined, params_3dmm, to_2d=True)

        return {
            OUTPUT_LANDMARKS_HEATMAP: heatmap,
            OUTPUT_DEPTH: depth,
            OUTPUT_REGION: region,
            OUTPUT_3DMM_PARAMS: params_3dmm,
            OUTPUT_2D_LANDMARKS: landmarks,
            OUTPUT_3D_VERTICES: vertices_3d,
            OUTPUT_2D_VERTICES: vertices_2d,
            OUTPUT_RESIDUAL_DEFORMATION: residual_deformation,
            OUTPUT_3D_VERTICES_REFINED: vertices_3d_refined,
            OUTPUT_2D_VERTICES_REFINED: vertices_2d_refined,
        }