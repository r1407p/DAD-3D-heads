import os
import json
from typing import Dict, Any
import numpy as np
import cv2
import torch

from model_training.data import FlameDataset
from model_training.data.config import (
    IMAGE_FILENAME_KEY,
    SAMPLE_INDEX_KEY,
    INPUT_IMAGE_KEY,
    INPUT_BBOX_KEY,
    INPUT_SIZE_KEY,
    TARGET_2D_LANDMARKS,
    TARGET_LANDMARKS_HEATMAP,
    TARGET_2D_LANDMARKS_PRESENCE,
    TARGET_2D_FULL_LANDMARKS,
    TARGET_FACE_REGION,
    TARGET_FACE_DEPTH,
)

def tensor_to_bgr_uint8(img_tensor: torch.Tensor, normalize_name: str = "imagenet") -> np.ndarray:
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # HWC, RGB
    if normalize_name == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img * std + mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif normalize_name in ("none", None):
        img = np.clip((img * 255.0) if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    else:
        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn + 1e-8)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def draw_points(image_bgr: np.ndarray, pts_xy: np.ndarray, color=(0, 0, 255), radius: int = None) -> np.ndarray:
    out = image_bgr.copy()
    H, W = out.shape[:2]
    rr = radius if radius is not None else max(1, int(min(H, W) * 0.005))
    for (x, y) in pts_xy.astype(int):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(out, (int(x), int(y)), rr, color, -1, lineType=cv2.LINE_AA)
    return out

def overlay_heatmap_on_image(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    hm = heatmap.max(axis=0) if heatmap.ndim == 3 else heatmap
    hm = hm.astype(np.float32)
    hm -= hm.min()
    hm = hm / (hm.max() + 1e-8)
    hm_u8 = (hm * 255.0).astype(np.uint8)
    H, W = image_bgr.shape[:2]
    hm_u8 = cv2.resize(hm_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0, hm_color, alpha, 0)

def visualize_dataset(train_config_path: str, output_dir: str, max_items: int = None):
    config_all = json.load(open(train_config_path, "r"))
    train_cfg: Dict[str, Any] = config_all["train"]
    dataset: FlameDataset = FlameDataset.from_config(config=train_cfg)

    os.makedirs(output_dir, exist_ok=True)

    norm_name = train_cfg.get("transform", {}).get("normalize", "imagenet")
    img_size = dataset.img_size

    for idx in range(len(dataset)):
        if max_items is not None and idx >= max_items:
            break

        item = dataset[idx]
        ann = dataset.data[idx]

        item_dir = os.path.join(output_dir, str(idx))
        os.makedirs(item_dir, exist_ok=True)

        img_bgr = tensor_to_bgr_uint8(item[INPUT_IMAGE_KEY], normalize_name=norm_name)
        H_vis, W_vis = img_bgr.shape[:2]

        lm_norm = item[TARGET_2D_LANDMARKS]
        lm_pix = (lm_norm * img_size).astype(np.float32)
        if TARGET_2D_LANDMARKS_PRESENCE in item:
            presence = item[TARGET_2D_LANDMARKS_PRESENCE].astype(bool)
            lm_pix = lm_pix[presence]
        img_kp = draw_points(img_bgr, lm_pix, color=(0, 0, 255))

        hm = item[TARGET_LANDMARKS_HEATMAP]
        img_hm = overlay_heatmap_on_image(img_bgr, hm, alpha=0.5)

        if TARGET_2D_FULL_LANDMARKS in item:
            verts2d_aug = item[TARGET_2D_FULL_LANDMARKS].astype(np.float32)
            img_proj = draw_points(img_bgr, verts2d_aug, color=(0, 255, 0), radius=1)
        else:
            img_proj = img_bgr.copy()

        if TARGET_FACE_DEPTH in item:
            depth01 = item[TARGET_FACE_DEPTH].detach().cpu().numpy().squeeze().astype(np.float32)
            depth_u8 = (np.clip(depth01, 0, 1) * 255.0).astype(np.uint8)
            depth_u8 = cv2.resize(depth_u8, (W_vis, H_vis), interpolation=cv2.INTER_LINEAR)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
            img_depth_overlay = cv2.addWeighted(img_bgr, 1.0, depth_color, 0.6, 0)
        else:
            depth_color = np.zeros_like(img_bgr)
            img_depth_overlay = img_bgr.copy()

        # region: [1,64,64] float32 in {0,1}
        if TARGET_FACE_REGION in item:
            region01 = item[TARGET_FACE_REGION].detach().cpu().numpy().squeeze().astype(np.float32)
            region_u8 = (np.clip(region01, 0, 1) * 255.0).astype(np.uint8)
            region_u8 = cv2.resize(region_u8, (W_vis, H_vis), interpolation=cv2.INTER_NEAREST)
            region_color = cv2.applyColorMap(region_u8, cv2.COLORMAP_BONE)
            img_region_overlay = cv2.addWeighted(img_bgr, 1.0, region_color, 0.5, 0)
        else:
            region_color = np.zeros_like(img_bgr)
            img_region_overlay = img_bgr.copy()

        # 6) 存檔
        cv2.imwrite(os.path.join(item_dir, "input.png"), img_bgr)
        cv2.imwrite(os.path.join(item_dir, "input_landmarks.png"), img_kp)
        cv2.imwrite(os.path.join(item_dir, "input_heatmap.png"), img_hm)
        cv2.imwrite(os.path.join(item_dir, "projected_vertices.png"), img_proj)
        cv2.imwrite(os.path.join(item_dir, "face_depth.png"), depth_color)
        cv2.imwrite(os.path.join(item_dir, "face_region.png"), region_color)
        cv2.imwrite(os.path.join(item_dir, "face_depth_overlay.png"), img_depth_overlay)
        cv2.imwrite(os.path.join(item_dir, "face_region_overlay.png"), img_region_overlay)

        x, y, w, h = item[INPUT_BBOX_KEY]
        meta = {
            "index": int(item[SAMPLE_INDEX_KEY]),
            "img_path": ann["img_path"],
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "image_size_input": [int(H_vis), int(W_vis)],
            "num_landmarks": int(lm_norm.shape[0]),
        }
        with open(os.path.join(item_dir, "data.json"), "w") as f:
            json.dump(meta, f, indent=2)

        orig = cv2.imread(os.path.join(dataset.config["dataset_root"], ann["img_path"]))
        if orig is not None:
            cv2.imwrite(os.path.join(item_dir, "original.png"), orig)

        print(f"[{idx+1}/{len(dataset)}] saved -> {item_dir}")
        cv2.imwrite(os.path.join(item_dir, "face_depth_raw.png"), depth_u8)
        cv2.imwrite(os.path.join(item_dir, "face_region_raw.png"), region_u8)
        breakpoint()
if __name__ == "__main__":
    train_config_path = "config.json"
    output_dir = "visualize/train_dataset"
    visualize_dataset(train_config_path, output_dir, max_items=None)


# python visualize_dataset.py