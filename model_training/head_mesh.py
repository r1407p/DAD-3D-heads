from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from model_training.model.flame import FLAMELayer, FLAME_CONSTS, FlameParams


class HeadMesh(nn.Module):
    def __init__(
        self,
        flame_config: Optional[Dict[str, int]] = None,
        batch_size: int = 1,
        image_size: int = 256,
    ):
        super().__init__()
        if flame_config is None:
            self.flame_constants = FLAME_CONSTS
        else:
            self.flame_constants = flame_config
        self.flame = FLAMELayer(consts=self.flame_constants, batch_size=batch_size)
        self._image_size = image_size

    def flame_params(self, params_3dmm: Tensor) -> FlameParams:
        flame_params = FlameParams.from_3dmm(params_3dmm, self.flame_constants)
        return flame_params

    def vertices_3d(self, params_3dmm: Tensor, zero_rotation: bool = False) -> Tensor:
        flame_params = self.flame_params(params_3dmm=params_3dmm)
        pred_vertices = self.flame.forward(flame_params, zero_rot=zero_rotation)
        return pred_vertices

    def reprojected_vertices(self, params_3dmm: Tensor, to_2d: bool = True) -> Tensor:
        """
        Returns [B, N, C].
        """
        flame_params = self.flame_params(params_3dmm=params_3dmm)
        pred_vertices = self.flame.forward(flame_params, zero_rot=False)
        scale = torch.clamp(flame_params.scale[:, None] + 1.0, 1e-8)
        pred_vertices *= scale  # [B, 1, 1]
        flame_params.translation[..., 2] = 0.0
        pred_vertices += flame_params.translation[:, None]  # [B, 1, 3]
        projected_vertices = (pred_vertices + 1.0) / 2.0 * self._image_size
        if to_2d:
            projected_vertices = projected_vertices[..., :2]
        return projected_vertices

    def reprojected_vertices_from_vertices_3d(
        self,
        vertices_3d: torch.Tensor,
        params_3dmm: torch.Tensor,
        to_2d: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            vertices_3d: [B, V, 3] refined or original vertices in FLAME camera-aligned space
            params_3dmm: [B, num_params] (used only for scale & translation)
        Returns:
            projected_vertices: [B, V, 2 or 3]
        """

        flame_params = self.flame_params(params_3dmm)
        vertices_3d = self.flame.to_rot(vertices_3d, flame_params)

        # 1. scale (same as original reproject)
        scale = torch.clamp(flame_params.scale[:, None] + 1.0, 1e-8)
        vertices = vertices_3d * scale  # [B, V, 3]

        # 2. translation (z forced to 0, same behavior)
        translation = flame_params.translation.clone()
        translation[..., 2] = 0.0
        vertices = vertices + translation[:, None, :]  # [B, V, 3]

        # 3. NDC [-1,1] → image coordinates
        projected_vertices = (vertices + 1.0) / 2.0 * self._image_size

        if to_2d:
            projected_vertices = projected_vertices[..., :2]

        return projected_vertices
    def adjust_3dmm_to_paddings(self, params_3dmm: Tensor, paddings: List[int]) -> Tensor:
        """
        paddings: if you enlarge the image, the paddings should be positive; if you crop it - negative.
        paddings = [pad_top, pad_bottom, pad_left, pad_right]
        """
        flame_params = self.flame_params(params_3dmm=params_3dmm)
        flame_params.translation = (
            flame_params.translation
            + Tensor([[paddings[2], paddings[0], 0]]).to(params_3dmm.device) * 2 / self._image_size
        )
        params_3dmm = flame_params.to_3dmm_tensor()

        return params_3dmm
