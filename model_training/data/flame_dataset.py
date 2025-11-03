import os
import json
from typing import Dict, Any, List, Union, Tuple, Optional
from collections import namedtuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
import pytorch_toolbelt.utils as pt_utils
from skimage.draw import polygon

from hydra.utils import instantiate

from model_training.data.config import (
    IMAGE_FILENAME_KEY,
    SAMPLE_INDEX_KEY,
    INPUT_IMAGE_KEY,
    INPUT_BBOX_KEY,
    INPUT_SIZE_KEY,
    TARGET_PROJECTION_MATRIX,
    TARGET_3D_MODEL_VERTICES,
    TARGET_3D_WORLD_VERTICES,
    TARGET_2D_LANDMARKS,
    TARGET_LANDMARKS_HEATMAP,
    TARGET_2D_FULL_LANDMARKS,
    TARGET_2D_LANDMARKS_PRESENCE,
)
# --- 盡量使用專案內的常數；若無則 fallback ---
try:
    from model_training.data.config import TARGET_FACE_REGION, TARGET_FACE_DEPTH
except Exception:
    TARGET_FACE_REGION = "TARGET_FACE_REGION"
    TARGET_FACE_DEPTH = "target_face_depth"

from model_training.data.transforms import get_resize_fn, get_normalize_fn
from model_training.data.utils import (
    ensure_bbox_boundaries,
    extend_bbox,
    read_as_rgb,
    get_68_landmarks,
)
from model_training.utils import load_2d_indices, create_logger

# 取靜態資源路徑（與 utils.py 一致）
from utils import get_relative_path

MeshArrays = namedtuple(
    "MeshArrays",
    ["vertices3d", "vertices3d_world_homo", "projection_matrix"],
)

logger = create_logger(__name__)


def collate_skip_none(batch: Any) -> Any:
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)


class FlameDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        self.data = data
        self.config = config

        self.img_size = config["img_size"]
        self.filename_key = "img_path"

        # 只做「幾何」的增廣（resize 等）；Normalize 我們稍後只套在 image
        self.aug_pipeline = self._get_aug_pipeline(config["transform"])
        self.normalize_t = get_normalize_fn(config["transform"].get("normalize", "imagenet"))

        self.num_classes = config.get("num_classes")
        self.keypoints_indices = load_2d_indices(config["keypoints"])

        # 影像會用 image_to_tensor；depth/mask 另外處理
        self.tensor_keys = [INPUT_IMAGE_KEY]
        self.coder = instantiate(config["coder"], config, self.num_classes)

        # 載入 FLAME faces 與臉部子集三角形
        self.mesh_faces = self._load_mesh_faces()
        self.face_vertex_set = self._get_face_vertex_set()
        self.face_faces = self._select_face_triangles(self.mesh_faces, self.face_vertex_set)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item_anno = self._get_item_anno(idx=idx)  # metadata of the image ex: bbox, image path, etc.
        item_data = self._parse_anno(item_anno)
        item_data = self._transform(item_data)
        item_dict = self._form_anno_dict(item_data)
        item_dict = self._add_index(idx, item_anno, item_dict)
        item_dict = self._convert_images_to_tensors(item_dict)
        return item_dict

    def _add_index(self, idx: int, annotation: Any, item_dict: Dict[str, Any]) -> Dict[str, Any]:
        if item_dict is not None:
            item_dict.update({SAMPLE_INDEX_KEY: idx, IMAGE_FILENAME_KEY: annotation[self.filename_key]})
        return item_dict

    def _get_item_anno(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        with open(config["ann_path"]) as json_file:
            anno = json.load(json_file)
        return cls(data=anno, config=config)

    # ------------------------- 核心：3D→2D + z-buffer 產 depth/mask -------------------------

    @staticmethod
    def _load_mesh_faces() -> np.ndarray:
        """載入 FLAME 的三角面（0-based），[F,3] int32。"""
        faces_path = get_relative_path("../model/static/flame_mesh_faces.pt", __file__)
        faces = torch.load(faces_path)
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        return faces.astype(np.int32)

    @staticmethod
    def _get_face_vertex_set() -> Optional[np.ndarray]:
        """
        優先讀臉部頂點集合（face_vertices.npy / face_verts.npy / face_indices.npy）；
        若不存在則退回用 face_edges.npy 端點集合近似；失敗回 None。
        """
        base_dir = get_relative_path("../model/static/flame_indices", __file__)
        candidates = ["face_vertices.npy", "face_verts.npy", "face_indices.npy"]
        for name in candidates:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                v = np.load(p)
                return np.unique(v.astype(np.int32))
        # fallback by edges
        p_edges = os.path.join(base_dir, "face_edges.npy")
        if os.path.exists(p_edges):
            edges = np.load(p_edges).astype(np.int32)  # [E,2]
            return np.unique(edges.reshape(-1))
        return None

    @staticmethod
    def _select_face_triangles(faces: np.ndarray, face_vs: Optional[np.ndarray]) -> np.ndarray:
        """只保留三個頂點都在臉部集合內的三角形；若 face_vs=None 則回傳全部 faces。"""
        if face_vs is None or face_vs.size == 0:
            return faces
        keep = np.isin(faces, face_vs).all(axis=1)
        return faces[keep]

    @staticmethod
    def _rasterize_depth_and_mask(
        xy_crop: np.ndarray,
        depth_per_vert: np.ndarray,
        faces: np.ndarray,
        crop_w: int,
        crop_h: int,
        valid_vert_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        簡單 z-buffer 光柵化：回傳 depth01 (float32, [0,1]) 與 mask (uint8, 0/1)，與裁切後座標系對齊。
        - xy_crop: [N,2]，已扣掉 (crop_x, crop_y)
        - depth_per_vert: [N]，越小越近（建議 -z_cam）
        - faces: [F,3] int32
        - valid_vert_mask: [N] 只渲染 w>0 的頂點/三角形
        """
        depth = np.full((crop_h, crop_w), np.inf, dtype=np.float32)
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

        if valid_vert_mask is None:
            valid_vert_mask = np.ones(len(xy_crop), dtype=bool)

        for (i, j, k) in faces:
            if not (valid_vert_mask[i] and valid_vert_mask[j] and valid_vert_mask[k]):
                continue

            tri = xy_crop[[i, j, k], :]  # [[x,y],[x,y],[x,y]]
            if not np.all(np.isfinite(tri)):
                continue

            # 以 skimage 取得三角形覆蓋像素（row=y, col=x）
            rr, cc = polygon(tri[:, 1], tri[:, 0], shape=(crop_h, crop_w))
            if rr.size == 0:
                continue

            # 重心座標（像素中心）
            ax, ay = tri[0]; bx, by = tri[1]; cx, cy = tri[2]
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-8:
                continue

            px = cc + 0.5
            py = rr + 0.5
            alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
            beta = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
            gamma = 1.0 - alpha - beta

            z_tri = depth_per_vert[[i, j, k]]  # [3]
            z_interp = alpha * z_tri[0] + beta * z_tri[1] + gamma * z_tri[2]

            cur = depth[rr, cc]
            closer = z_interp < cur
            if np.any(closer):
                depth[rr[closer], cc[closer]] = z_interp[closer]
                mask[rr[closer], cc[closer]] = 1  # 0/1

        # 以有效區域的 min/max 做 [0,1] 正規化
        valid = (mask > 0) & np.isfinite(depth)
        depth01 = np.zeros_like(depth, dtype=np.float32)
        if np.any(valid):
            dmin = float(depth[valid].min())
            dmax = float(depth[valid].max())
            denom = max(dmax - dmin, 1e-8)
            depth01[valid] = (depth[valid] - dmin) / denom

        # resize to [64 * 64]
        depth01 = cv2.resize(depth01, (64, 64), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)
        return depth01, mask

    # --------------------------------------------------------------------------------------

    def _convert_images_to_tensors(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        if item_data is not None:
            for key, item in item_data.items():
                if isinstance(item, np.ndarray) and key in self.tensor_keys:
                    item_data[key] = pt_utils.image_to_tensor(item.astype("float32"))
                if isinstance(item, np.ndarray) and key in [TARGET_FACE_DEPTH, TARGET_FACE_REGION]:
                    item_data[key] = torch.from_numpy(item.astype("float32"))
        return item_data

    def _parse_anno(self, item_anno: Dict[str, Any]) -> Dict[str, Any]:
        img = read_as_rgb(os.path.join(self.config["dataset_root"], item_anno["img_path"]))

        # 隨機擴 bbox（每邊 5%~15%） + 裁切
        bbox = item_anno["bbox"]
        offset = tuple(0.1 * np.random.uniform(size=4) + 0.05)
        x, y, w, h = ensure_bbox_boundaries(extend_bbox(np.array(bbox), offset), img.shape[:2])
        cropped_img = img[y : y + h, x : x + w]

        # 載入 mesh & 相機
        flame_vertices3d, flame_vertices3d_world_homo, projection_matrix = self._load_mesh(
            os.path.join(self.config["dataset_root"], item_anno["annotation_path"])
        )

        # --- 3D→2D 投影（裁切座標系） ---
        # 供 landmarks 使用的 2D 投影會在 _get_2d_landmarks_w_presence 裡做
        # 這裡額外為 depth/mask 準備「全部頂點」的 2D 投影與 z/depth 與 w-valid
        # 1) 以原圖高做 y-flip，再扣裁切原點
        vertices2d_homo = (projection_matrix @ flame_vertices3d_world_homo.T).T  # [N,4]
        w_clip = vertices2d_homo[:, 3:4]
        valid_w = (w_clip[:, 0] > 1e-8)

        xy = vertices2d_homo[:, :2] / np.clip(w_clip, 1e-8, None)
        xy = np.stack([xy[:, 0], (img.shape[0] - xy[:, 1])], axis=1)  # 轉影像座標（左上原點）
        xy_crop = xy - np.array([x, y], dtype=np.float32)             # 對齊裁切座標系 [0..w), [0..h)

        # 2) 深度（相機多為 -Z 朝前，取 -z 讓「越小越近」→ z-buffer 用 min）
        z_cam = flame_vertices3d_world_homo[:, 2] / np.clip(flame_vertices3d_world_homo[:, 3], 1e-8, None)
        depth_per_vert = -z_cam if np.median(z_cam) < 0 else z_cam

        # 3) 只用臉部三角形做 z-buffer（fallback: 全部 faces）
        faces = self.face_faces if self.face_faces is not None and len(self.face_faces) > 0 else self.mesh_faces
        depth01, face_mask = self._rasterize_depth_and_mask(
            xy_crop.astype(np.float32), depth_per_vert.astype(np.float32), faces, w, h, valid_vert_mask=valid_w
        )

        return {
            INPUT_IMAGE_KEY: cropped_img,
            INPUT_BBOX_KEY: (x, y, w, h),
            INPUT_SIZE_KEY: img.shape,
            TARGET_3D_MODEL_VERTICES: flame_vertices3d,
            TARGET_3D_WORLD_VERTICES: flame_vertices3d_world_homo,
            TARGET_PROJECTION_MATRIX: projection_matrix,
            TARGET_FACE_DEPTH: depth01,       # [h, w] float32 0~1
            TARGET_FACE_REGION: face_mask,      # [h, w] uint8 0/1
        }

    @staticmethod
    def _load_mesh(mesh_path: str) -> MeshArrays:
        with open(mesh_path) as json_data:
            data = json.load(json_data)
        flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
        model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
        flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
        # rotated and translated (to world/camera coordinates)
        flame_vertices3d_world_homo = (model_view_matrix @ flame_vertices3d_homo.T).T
        return MeshArrays(
            vertices3d=flame_vertices3d,
            vertices3d_world_homo=flame_vertices3d_world_homo,  # with pose and translation
            projection_matrix=np.array(data["projection_matrix"], dtype=np.float32),
        )

    @staticmethod
    def _project_vertices_onto_image(
        vertices3d_world_homo: np.ndarray,
        projection_matrix: np.ndarray,
        height: int,
        crop_point_x: int,
        crop_point_y: int,
    ):
        vertices2d_homo = (projection_matrix @ vertices3d_world_homo.T).T
        vertices2d = vertices2d_homo[:, :2] / vertices2d_homo[:, [3]]
        vertices2d = np.stack((vertices2d[:, 0], (height - vertices2d[:, 1])), -1)
        vertices2d -= (crop_point_x, crop_point_y)
        return vertices2d

    def _get_2d_landmarks_w_presence(
        self,
        vertices3d_world_homo: np.ndarray,
        projection_matrix: np.ndarray,
        img_shape: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.num_classes == 68:
            landmarks_3d_world_subset = get_68_landmarks(
                torch.from_numpy(vertices3d_world_homo[..., :3]).view(-1, 3)
            ).numpy()
            landmarks_3d_world_subset = np.concatenate(
                (landmarks_3d_world_subset, np.ones_like(landmarks_3d_world_subset[:, [0]])), -1
            )
        else:
            landmarks_3d_world_subset = vertices3d_world_homo[self.keypoints_indices]
        x, y, w, h = bbox

        landmarks_2d_subset = self._project_vertices_onto_image(
            landmarks_3d_world_subset, projection_matrix, img_shape[0], x, y
        )
        keypoints_2d = self._project_vertices_onto_image(vertices3d_world_homo, projection_matrix, img_shape[0], x, y)

        presence_subset = np.array([False] * len(landmarks_2d_subset))
        for i in range(len(landmarks_2d_subset)):
            if 0 < landmarks_2d_subset[i, 0] < w and 0 < landmarks_2d_subset[i, 1] < h:
                presence_subset[i] = True
        return landmarks_2d_subset, presence_subset, keypoints_2d

    def _transform(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        vertices_2d_subset, presence_subset, vertices_2d = self._get_2d_landmarks_w_presence(
            item_data[TARGET_3D_WORLD_VERTICES],
            item_data[TARGET_PROJECTION_MATRIX],
            item_data[INPUT_SIZE_KEY],
            item_data[INPUT_BBOX_KEY],
        )

        # 幾何增廣（resize 等）同步作用於 image / keypoints / depth / mask
        result = self.aug_pipeline(
            image=item_data[INPUT_IMAGE_KEY],
            keypoints=np.concatenate((vertices_2d_subset, vertices_2d), 0),
            **{
                TARGET_FACE_DEPTH: item_data[TARGET_FACE_DEPTH],
                TARGET_FACE_REGION: item_data[TARGET_FACE_REGION],
            }
        )

        # 僅對 image 做 Normalize
        # result_img = self.normalize_t(image=result["image"])["image"]

        return {
            INPUT_IMAGE_KEY: result["image"],
            INPUT_BBOX_KEY: item_data[INPUT_BBOX_KEY],
            TARGET_3D_MODEL_VERTICES: item_data[TARGET_3D_MODEL_VERTICES],
            TARGET_2D_LANDMARKS: np.array(result["keypoints"][: self.num_classes], dtype=np.float32),
            TARGET_2D_FULL_LANDMARKS: np.array(result["keypoints"][self.num_classes :], dtype=np.float32),
            TARGET_2D_LANDMARKS_PRESENCE: presence_subset,
            TARGET_FACE_DEPTH: result[TARGET_FACE_DEPTH].astype(np.float32),   # [h',w']
            TARGET_FACE_REGION: (result[TARGET_FACE_REGION].astype(np.uint8)),     # [h',w'] 0/1
        }

    def _form_anno_dict(self, item_data: Dict[str, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        landmarks = item_data[TARGET_2D_LANDMARKS]
        presence = item_data[TARGET_2D_LANDMARKS_PRESENCE]
        heatmap = self.coder(landmarks, presence)

        # 這裡維持你原本的正規化（除以單一 img_size）；若要更嚴謹可改為除以 (W', H')
        item_data[TARGET_2D_LANDMARKS] = landmarks / self.img_size
        item_data[TARGET_LANDMARKS_HEATMAP] = np.uint8(255.0 * heatmap)

        return item_data

    def _get_aug_pipeline(self, aug_config: Dict[str, Any]) -> A.Compose:
        """
        幾何增廣：只做 resize（LongestMaxSize），Normalize 另外在 _transform 對 image 單獨套用。
        讓 depth 用 'image' 類型（雙線性），mask 用 'mask' 類型（最近鄰）。
        """
        resize = get_resize_fn(self.img_size, mode=aug_config.get("resize_mode", "longest_max_size"))
        return A.Compose(
            [resize],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            additional_targets={
                TARGET_FACE_DEPTH: "image",  # 雙線性
                TARGET_FACE_REGION: "mask",    # 最近鄰
            },
        )

    def get_collate_fn(self) -> Any:
        return collate_skip_none
