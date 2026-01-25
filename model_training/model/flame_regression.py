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
import os
import numpy as np
from typing import Optional
from model_training.model.SLPT import get_roi, interpolation_layer


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


class SLPTDeformationHead(nn.Module):
    def __init__(self, num_vertices, sample_num, img_size, feature_channels, intermediate_channels, hidden_dim, scale=0.01):
        super().__init__()
        self.num_vertices = num_vertices
        self.sample_num = sample_num
        self.img_size = img_size
        self.feature_channels = feature_channels
        self.intermediate_channels = intermediate_channels
        self.hidden_dim = hidden_dim
        self.scale = scale

        self.upscaling_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.feature_channels, self.hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 3, padding=1),
        )

        self.f_head = nn.Conv2d(self.hidden_dim + self.intermediate_channels, self.hidden_dim, 3, padding=1)
        
        self.ROI = get_roi(self.sample_num, 8.0, 64)
        self.interpolation = interpolation_layer()
        self.feature_extractor = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=self.sample_num, bias=False)
        self.projection = nn.Linear(self.hidden_dim+3, 3)

    def forward(self, feature, intermediate_feature, vertices_2d, original_vertices_3d):
        B = vertices_2d.shape[0]
        # upscaling feature
        upscaling_feature = self.upscaling_head(feature)
        input_feature = torch.cat([upscaling_feature, intermediate_feature], dim=1)
        input_feature = self.f_head(input_feature)

        #  ROI_features
        vertices_2d = vertices_2d / 256
        ROI_anchor, bbox_size, start_anchor = self.ROI(vertices_2d.detach())
        ROI_anchor = ROI_anchor.view(B, self.num_vertices * self.sample_num * self.sample_num, 2)
        ROI_feature = self.interpolation(input_feature, ROI_anchor.detach()).view(B, self.num_vertices, self.sample_num, self.sample_num, self.hidden_dim)
        ROI_feature = ROI_feature.view(B * self.num_vertices, self.sample_num, self.sample_num, self.hidden_dim).permute(0, 3, 2, 1)

        transformed_feature = self.feature_extractor(ROI_feature).view(B, self.num_vertices, self.hidden_dim)
        transformed_feature = torch.cat([transformed_feature, original_vertices_3d], dim=2)
        deformation = self.projection(transformed_feature)
        return deformation


class FlameRegression(nn.Module):
    def __init__(self, model_config: Dict[str, Any], consts_config: Dict[str, Any], flame_indices_config: Dict[str, Any], num_classes: int = 68, only_face: bool = False):
        super().__init__()
        self.flame_indices_config = flame_indices_config
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

        self.flame_indices = {}
        for key, value in self.flame_indices_config["files"].items():
            self.flame_indices[key] = np.load(os.path.join(self.flame_indices_config["folder"], value))

        self.only_face = only_face
        if only_face:
            num_vertices = len(self.flame_indices['face'])
            self.refined_indices = self.flame_indices['face']
        else:
            num_vertices = 5023  # FLAME vertex count
            self.refined_indices = np.arange(5023)
        self.num_vertices = num_vertices
        self.flame_faces = torch.load('model_training/model/static/flame_mesh_faces.pt')

        self.deformation_type = "SLPT"  # "SLPT" or "None"

        if self.deformation_type == "MLP":
            pre_residual_head_in_dim = model_config["num_filters"] + model_config["num_classes"] + 1 + 1 + self.encoder.encoder_channels["layer1"]
            self.pre_residual_head = nn.Conv2d(pre_residual_head_in_dim, 256, kernel_size=1)
            self.residual_head = ResidualDeformationHead(
                in_dim=2048,
                num_vertices=num_vertices,
                hidden_dim=256,
                scale=0.01
            )
        elif self.deformation_type == "SLPT":
            self.slpt_deformation_head = SLPTDeformationHead(
                num_vertices=num_vertices,
                sample_num=5,
                img_size=self._img_size,
                feature_channels=model_config["num_filters"] + self.encoder.encoder_channels["layer1"],
                intermediate_channels=model_config["num_classes"] + 1 + 1,
                hidden_dim=256,
                scale=0.01
            )
            self.Sample_num = 5
            pre_residual_head_in_dim = model_config["num_filters"] + model_config["num_classes"] + 1 + 1 + self.encoder.encoder_channels["layer1"]
            self.conv1 = nn.Conv2d(pre_residual_head_in_dim, 256, kernel_size=1)
            self.ROI_1 = get_roi(self.Sample_num, 8.0, 64)
            self.interpolation = interpolation_layer()
            self.feature_extractor = nn.Conv2d(256, 256, kernel_size=self.Sample_num, bias=False)
            self.projection = nn.Linear(256+3, 3)
        

    def compute_vertex_visibility_torch(
        self,
        vertices_3d: torch.Tensor,
        camera_position: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batched back-face culling vertex visibility (PyTorch).

        Args:
            vertices_3d: (B, N, 3)
            faces: (F, 3) long tensor
            camera_position:
                - None: orthographic camera looking along -Z
                - (3,) or (B, 3): perspective camera position

        Returns:
            visibility: (B, N) bool tensor
        """
        assert vertices_3d.dim() == 3 and vertices_3d.size(-1) == 3
        assert self.flame_faces.dim() == 2 and self.flame_faces.size(1) == 3

        B, N, _ = vertices_3d.shape
        F = self.flame_faces.shape[0]

        # --- Gather face vertices ---
        v0 = vertices_3d[:, self.flame_faces[:, 0]]  # (B, F, 3)
        v1 = vertices_3d[:, self.flame_faces[:, 1]]
        v2 = vertices_3d[:, self.flame_faces[:, 2]]
        # --- Face normals ---
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, F, 3)
        norm = torch.linalg.norm(face_normals, dim=-1, keepdim=True)  # (B, F, 1)

        valid = norm > 1e-8
        face_normals = torch.where(
            valid,
            face_normals / (norm + 1e-8),
            torch.zeros_like(face_normals)
        )

        # --- View direction ---
        if camera_position is None:
            # Orthographic: constant view direction
            view_dirs = torch.tensor(
                [0.0, 0.0, -1.0],
                dtype=vertices_3d.dtype
            ).view(1, 1, 3).expand(B, F, 3).to(vertices_3d.device)
        else:
            if camera_position.dim() == 1:
                camera_position = camera_position.view(1, 1, 3).expand(B, F, 3).to(vertices_3d.device)
            elif camera_position.dim() == 2:
                camera_position = camera_position.view(B, 1, 3).expand(B, F, 3).to(vertices_3d.device)
            else:
                raise ValueError("camera_position must be (3,) or (B, 3)")

            face_centers = (v0 + v1 + v2) / 3.0
            view_dirs = camera_position - face_centers
            view_dirs = view_dirs / (torch.linalg.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
        # --- Front-facing test ---
        facing = (face_normals * view_dirs).sum(dim=-1) > 0  # (B, F)

        # --- Accumulate vertex visibility ---
        visibility = torch.zeros(B, N, dtype=torch.bool)

        # faces: (F, 3) → broadcast to (B, F, 3)
        faces_expand = self.flame_faces.unsqueeze(0).expand(B, -1, -1)

        # Mark vertices belonging to any visible face
        visibility = torch.zeros(B, N, dtype=torch.bool, device=vertices_3d.device)

        for b in range(B):
            visible_faces = self.flame_faces[facing[b]]        # (F_visible, 3)
            visible_vertices = visible_faces.reshape(-1)
            visibility[b, visible_vertices] = True

        return visibility

    def forward(self, x):
        encoder_output = [] # [(bs, 64, 64, 64), (bs, 256, 64, 64), (bs, 512, 32, 32), (bs, 1024, 16, 16)]
        for stage in self.encoder.stages[: self.max_layer]:
            x = stage(x)
            encoder_output.append(x)
        decoder_output = self.bifpn(encoder_output[1:]) # [(bs, 256, 64, 64), (bs, 256, 32, 32), (bs, 256, 16, 16), (bs, 256, 8, 8), (bs, 256, 4, 4)]
        heatmap = self.head(decoder_output) # (bs, 68, 64, 64)
        depth = self.depth(decoder_output) # (bs, 1, 64, 64)
        region = self.region(decoder_output) # (bs, 1, 64, 64)
        fmap = self.fusion_layer(x, heatmap, depth, region, decoder_output[2]) # (bs, 1024, 16, 16)
        fmap = self.encoder.stages[-1](fmap) # (bs, 2048, 8, 8)
        shape = self.shape(fmap).tanh() * self.limit_value # (bs, 403)
        pose = self.pose(fmap) # (bs, 10)
        landmarks = self.landmarks(fmap) # (bs, 136)
        B, N = landmarks.size()
        landmarks = F.relu(landmarks.reshape((B, N // 2, 2)), inplace=True) # (bs, 68, 2)

        params_3dmm = torch.cat([shape, pose], dim=1) # (bs, 413)
        
        # Compute 3D vertices and 2D reprojected vertices from 3DMM params
        vertices_3d = self.head_mesh.vertices_3d(params_3dmm=params_3dmm, zero_rotation=True)
        vertices_2d = self.head_mesh.reprojected_vertices(params_3dmm=params_3dmm, to_2d=True)

        rotated_vertices_3d = self.head_mesh.flame.to_rot(vertices_3d, self.head_mesh.flame_params(params_3dmm))
        visible_vertices = self.compute_vertex_visibility_torch(rotated_vertices_3d, None)

        vertices_3d_refined = vertices_3d.clone()
        if self.deformation_type == "MLP":
            res_feat = F.adaptive_avg_pool2d(fmap, 1).view(B, -1)
            residual_deformation = self.residual_head(res_feat)
            vertices_3d_refined[:, self.refined_indices] = (
                vertices_3d_refined[:, self.refined_indices] + residual_deformation
            )
            pass
        elif self.deformation_type == "SLPT":
            residual_deformation = self.slpt_deformation_head(torch.cat([x, decoder_output[2]], dim=1), torch.cat([heatmap, depth, region], dim=1), vertices_2d[:, self.refined_indices, :], vertices_3d[:, self.refined_indices, :])
            vertices_3d_refined[:, self.refined_indices] = (
                vertices_3d_refined[:, self.refined_indices] + residual_deformation
            )
            pass
        elif self.deformation_type == "None":
            residual_deformation = torch.zeros_like(vertices_3d_refined[:, self.refined_indices])
            pass

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
            # 'OUTPUT_VISIBLE_VERTICES': visible_vertices,
        }