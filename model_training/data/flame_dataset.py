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
    TARGET_FACE_REGION,
    TARGET_FACE_DEPTH,
)
from model_training.data.transforms import get_resize_fn, get_normalize_fn
from model_training.data.utils import (
    ensure_bbox_boundaries,
    extend_bbox,
    read_as_rgb,
    get_68_landmarks,
)
from model_training.utils import load_2d_indices, create_logger

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

        self.use_precomputed_maps = bool(config.get("use_precomputed_maps", True))

        resize = get_resize_fn(self.img_size, mode=config["transform"].get("resize_mode", "longest_max_size"))
        self.aug_geom = A.ReplayCompose([resize], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
        self.normalize_t = get_normalize_fn(config["transform"].get("normalize", "imagenet"))

        self.aug_pipeline = self._get_aug_pipeline(config["transform"])

        self.num_classes = config.get("num_classes")
        self.keypoints_indices = load_2d_indices(config["keypoints"])

        self.tensor_keys = [INPUT_IMAGE_KEY]
        self.coder = instantiate(config["coder"], config, self.num_classes)

        self.mesh_faces = self._load_mesh_faces()
        self.face_vertex_set = self._get_face_vertex_set()
        self.face_faces = self._select_face_triangles(self.mesh_faces, self.face_vertex_set)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item_anno = self._get_item_anno(idx=idx)  # metadata of the image ex: bbox, image path, etc.
        original_item_data = self._parse_anno(item_anno)
        item_data = self._transform(original_item_data)
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


    @staticmethod
    def _load_mesh_faces() -> np.ndarray:
        faces_path = get_relative_path("../model/static/flame_mesh_faces.pt", __file__)
        faces = torch.load(faces_path)
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        return faces.astype(np.int32)

    @staticmethod
    def _get_face_vertex_set() -> Optional[np.ndarray]:
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
        if face_vs is None or face_vs.size == 0:
            return faces
        keep = np.isin(faces, face_vs).all(axis=1)
        return faces[keep]

    def _resolve_item_path(self, raw_path: str, kind: str) -> Optional[str]:
        ds_root = os.path.normpath(self.config["dataset_root"])
        ds_name = os.path.basename(ds_root)
        rp = (raw_path or "").replace("\\", "/")

        cands = []
        if os.path.isabs(rp):
            cands.append(rp)
        cands.append(os.path.join(ds_root, rp))
        if rp.startswith(ds_name + "/"):
            stripped = rp[len(ds_name) + 1 :]
            cands.append(os.path.join(ds_root, stripped))
        base = os.path.basename(rp)
        for split in ["train", "val", "test"]:
            cands.append(os.path.join(ds_root, split, kind, base))
        for split in ["train/", "val/", "test/"]:
            if rp.startswith(split):
                cands.append(os.path.join(ds_root, rp))

        for p in cands:
            if p and os.path.exists(p):
                return os.path.normpath(p)
        return None

    @staticmethod
    def _rasterize_depth_and_mask(
        xy_crop: np.ndarray,
        depth_per_vert: np.ndarray,
        faces: np.ndarray,
        crop_w: int,
        crop_h: int,
        valid_vert_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

            rr, cc = polygon(tri[:, 1], tri[:, 0], shape=(crop_h, crop_w))
            if rr.size == 0:
                continue

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

    def _convert_images_to_tensors(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        if item_data is not None:
            for key, item in item_data.items():
                if isinstance(item, np.ndarray) and key in self.tensor_keys:
                    item_data[key] = pt_utils.image_to_tensor(item.astype("float32"))
        for k in (TARGET_FACE_DEPTH, TARGET_FACE_REGION):
            if k in item_data and isinstance(item_data[k], np.ndarray):
                arr = item_data[k].astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                item_data[k] = torch.from_numpy(arr)
        return item_data

        return item_data

    def _parse_anno(self, item_anno: Dict[str, Any]) -> Dict[str, Any]:
        img_path = self._resolve_item_path(item_anno["img_path"], kind="images") \
                   or os.path.join(self.config["dataset_root"], item_anno["img_path"])
        mesh_path = self._resolve_item_path(item_anno["annotation_path"], kind="annotations") \
                    or os.path.join(self.config["dataset_root"], item_anno["annotation_path"])

        img = read_as_rgb(img_path)
        bbox = item_anno["bbox"]
        offset = tuple(0.1 * np.random.uniform(size=4) + 0.05)
        x, y, w, h = ensure_bbox_boundaries(extend_bbox(np.array(bbox), offset), img.shape[:2])
        cropped_img = img[y : y + h, x : x + w]

        flame_vertices3d, flame_vertices3d_world_homo, projection_matrix = self._load_mesh(mesh_path)

        pre_depth_crop = None
        pre_region_crop = None
        if self.use_precomputed_maps:
            images_dir = os.path.dirname(img_path)           # .../<split>/images
            split_dir = os.path.dirname(images_dir)          # .../<split>
            base = os.path.splitext(os.path.basename(img_path))[0]
            depth_path = os.path.join(split_dir, "depth_map",  base + ".png")
            region_path = os.path.join(split_dir, "regionmap", base + ".png")

            if not os.path.exists(depth_path):
                alt = depth_path.replace("DAD-3DHeadsDataset" + os.sep, "")
                if os.path.exists(alt):
                    depth_path = alt
            if not os.path.exists(region_path):
                alt = region_path.replace("DAD-3DHeadsDataset" + os.sep, "")
                if os.path.exists(alt):
                    region_path = alt

            if os.path.exists(depth_path) and os.path.exists(region_path):
                d_full = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                r_full = cv2.imread(region_path, cv2.IMREAD_GRAYSCALE)
                if d_full is not None and r_full is not None:
                    if d_full.dtype == np.uint16:
                        d_full = d_full.astype(np.float32) / 65535.0
                    elif d_full.dtype == np.uint8:
                        d_full = d_full.astype(np.float32) / 255.0
                    else:
                        d_full = d_full.astype(np.float32)  # 已是 float 的情況
                    r_full = (r_full > 127).astype(np.float32)

                    pre_depth_crop  = d_full[y : y + h, x : x + w]
                    pre_region_crop = r_full[y : y + h, x : x + w]

        return {
            INPUT_IMAGE_KEY: cropped_img,
            INPUT_BBOX_KEY: (x, y, w, h),
            INPUT_SIZE_KEY: img.shape,
            TARGET_3D_MODEL_VERTICES: flame_vertices3d,
            TARGET_3D_WORLD_VERTICES: flame_vertices3d_world_homo,
            TARGET_PROJECTION_MATRIX: projection_matrix,
            "PRE_DEPTH_CROP": pre_depth_crop,
            "PRE_REGION_CROP": pre_region_crop,
        }

    @staticmethod
    def _load_mesh(mesh_path: str) -> MeshArrays:
        with open(mesh_path) as json_data:
            data = json.load(json_data)
        flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
        model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
        flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
        # rotated and translated (to world coordinates)
        flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))
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
            crop_point_y: int
    ):
        vertices2d_homo = np.transpose(np.matmul(projection_matrix, np.transpose(vertices3d_world_homo)))
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

        pre_depth = item_data.get("PRE_DEPTH_CROP", None)
        pre_region = item_data.get("PRE_REGION_CROP", None)
        use_pre = self.use_precomputed_maps and (pre_depth is not None) and (pre_region is not None)

        if use_pre:
            geom_out = self.aug_geom(
                image=item_data[INPUT_IMAGE_KEY],
                keypoints=np.concatenate((vertices_2d_subset, vertices_2d), 0)
            )
            img_geom = geom_out["image"]
            keypoints_aug = geom_out["keypoints"]
            verts2d_aug = np.array(keypoints_aug[self.num_classes:], dtype=np.float32)

            maps_out = A.ReplayCompose.replay(
                geom_out["replay"],
                image=item_data["PRE_DEPTH_CROP"].astype(np.float32),
                mask=item_data["PRE_REGION_CROP"].astype(np.uint8),
                keypoints=np.concatenate((vertices_2d_subset, vertices_2d), 0)
            )
            depth_aug  = maps_out["image"].astype(np.float32)        # [H_aug, W_aug], 0~1
            region_aug = maps_out["mask"].astype(np.float32)         # 0/1

            norm_out = self.normalize_t(image=img_geom)
            img_aug = norm_out["image"]

            H_aug, W_aug = img_aug.shape[:2]
            depth01 = cv2.resize(depth_aug,  (64, 64), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            region01 = cv2.resize(region_aug, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.float32)

            return {
                INPUT_IMAGE_KEY: img_aug,
                INPUT_BBOX_KEY: item_data[INPUT_BBOX_KEY],
                TARGET_3D_MODEL_VERTICES: item_data[TARGET_3D_MODEL_VERTICES],
                TARGET_2D_LANDMARKS: np.array(keypoints_aug[: self.num_classes], dtype=np.float32),
                TARGET_2D_FULL_LANDMARKS: np.array(keypoints_aug[self.num_classes :], dtype=np.float32),
                TARGET_2D_LANDMARKS_PRESENCE: presence_subset,
                INPUT_SIZE_KEY: item_data[INPUT_SIZE_KEY],
                TARGET_FACE_DEPTH: depth01,    # [64,64], 0~1
                TARGET_FACE_REGION: region01,  # [64,64], 0/1
            }

        result = self.aug_pipeline(
            image=item_data[INPUT_IMAGE_KEY], keypoints=np.concatenate((vertices_2d_subset, vertices_2d), 0)
        )
        verts2d_aug = np.array(result["keypoints"][self.num_classes:], dtype=np.float32)

        verts3d_world_homo = item_data[TARGET_3D_WORLD_VERTICES]  # [N,4]
        z = verts3d_world_homo[:, 2].astype(np.float32)
        depth_per_vert = -z if np.median(z) < 0 else z

        faces = self.face_faces if (self.face_faces is not None and self.face_faces.size > 0) else self.mesh_faces

        H_aug, W_aug = result["image"].shape[:2]
        valid = np.isfinite(verts2d_aug).all(axis=1)
        depth01, region01 = self._rasterize_depth_and_mask(
            xy_crop=verts2d_aug,
            depth_per_vert=depth_per_vert,
            faces=faces,
            crop_w=W_aug,
            crop_h=H_aug,
            valid_vert_mask=valid,
        )

        return {
            INPUT_IMAGE_KEY: result["image"],
            INPUT_BBOX_KEY: item_data[INPUT_BBOX_KEY],
            TARGET_3D_MODEL_VERTICES: item_data[TARGET_3D_MODEL_VERTICES],
            TARGET_2D_LANDMARKS: np.array(result["keypoints"][: self.num_classes], dtype=np.float32),
            TARGET_2D_FULL_LANDMARKS: np.array(result["keypoints"][self.num_classes :], dtype=np.float32),
            TARGET_2D_LANDMARKS_PRESENCE: presence_subset,
            INPUT_SIZE_KEY: item_data[INPUT_SIZE_KEY],
            TARGET_FACE_DEPTH: depth01.astype(np.float32),   # [64,64], 0~1
            TARGET_FACE_REGION: region01.astype(np.float32), # [64,64], 0/1
        }

    def _form_anno_dict(self, item_data: Dict[str, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        landmarks = item_data[TARGET_2D_LANDMARKS]
        presence = item_data[TARGET_2D_LANDMARKS_PRESENCE]
        heatmap = self.coder(landmarks, presence)
        item_data[TARGET_2D_LANDMARKS] = landmarks / self.img_size
        item_data[TARGET_LANDMARKS_HEATMAP] = np.uint8(255.0 * heatmap)
        return item_data

    def _get_aug_pipeline(self, aug_config: Dict[str, Any]) -> A.Compose:
        normalize = get_normalize_fn(aug_config.get("normalize", "imagenet"))
        resize = get_resize_fn(self.img_size, mode=aug_config.get("resize_mode", "longest_max_size"))
        return A.Compose(
            [resize, normalize],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
        )

    def get_collate_fn(self) -> Any:
        return collate_skip_none
