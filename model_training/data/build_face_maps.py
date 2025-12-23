#!/usr/bin/env python3
import os
import json
import argparse
from typing import Tuple, Optional, List

import numpy as np
import torch
import cv2
from skimage.draw import polygon

# 專案內工具：mesh faces 與臉部子集
from utils import get_relative_path


# ---------- FLAME 靜態資源讀取 ----------

def load_flame_faces_zero_based() -> np.ndarray:
    faces_path = get_relative_path("../model/static/flame_mesh_faces.pt", __file__)
    faces = torch.load(faces_path)
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    return faces.astype(np.int32)


def load_face_vertex_set() -> Optional[np.ndarray]:
    base = get_relative_path("../model/static/flame_indices", __file__)
    for name in ["face_vertices.npy", "face_verts.npy", "face_indices.npy"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            v = np.load(p)
            return np.unique(v.astype(np.int32))
    # fallback：用邊的端點集合近似
    edges_p = os.path.join(base, "face_edges.npy")
    if os.path.exists(edges_p):
        e = np.load(edges_p).astype(np.int32)  # [E,2]
        return np.unique(e.reshape(-1))
    return None


def select_face_triangles(faces: np.ndarray, face_vs: Optional[np.ndarray]) -> np.ndarray:
    if face_vs is None or face_vs.size == 0:
        return faces
    keep = np.isin(faces, face_vs).all(axis=1)
    return faces[keep]


# ---------- 幾何：投影 + z-buffer 光柵化 ----------

def project_all_vertices_full(Vh: np.ndarray, P: np.ndarray, H: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳 xy_full[N,2]（影像座標，左上原點）、depth_per_vert[N]（越小越近）、valid_w[N]。"""
    Vh = Vh.astype(np.float32)
    P = P.astype(np.float32)
    clip = (P @ Vh.T).T  # [N,4]
    w = clip[:, 3:4]
    valid_w = (w[:, 0] > 1e-8)

    xy = clip[:, :2] / np.clip(w, 1e-8, None)
    xy = np.stack([xy[:, 0], (H - xy[:, 1])], axis=1)

    z_cam = Vh[:, 2] / np.clip(Vh[:, 3], 1e-8, None)
    depth = -z_cam if np.median(z_cam) < 0 else z_cam
    return xy.astype(np.float32), depth.astype(np.float32), valid_w


def rasterize_depth01_and_mask(
    xy: np.ndarray, depth_per_vert: np.ndarray, faces: np.ndarray, W: int, H: int, valid_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """z-buffer 光柵化出 depth01(float32[H,W]) 與 region(uint8[H,W])。"""
    depth = np.full((H, W), np.inf, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)

    for (i, j, k) in faces:
        if not (valid_v[i] and valid_v[j] and valid_v[k]):
            continue
        tri = xy[[i, j, k], :]
        if not np.all(np.isfinite(tri)):
            continue

        rr, cc = polygon(tri[:, 1], tri[:, 0], shape=(H, W))
        if rr.size == 0:
            continue

        ax, ay = tri[0]; bx, by = tri[1]; cx, cy = tri[2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-8:
            continue

        px = cc + 0.5; py = rr + 0.5
        alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        beta  = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        gamma = 1.0 - alpha - beta

        z_tri = depth_per_vert[[i, j, k]]
        z_interp = alpha * z_tri[0] + beta * z_tri[1] + gamma * z_tri[2]

        cur = depth[rr, cc]
        closer = z_interp < cur
        if np.any(closer):
            depth[rr[closer], cc[closer]] = z_interp[closer]
            mask[rr[closer], cc[closer]] = 1

    # 以有效臉部區域 min/max 正規化到 [0,1]
    valid = (mask > 0) & np.isfinite(depth)
    depth01 = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        dmin = float(depth[valid].min())
        dmax = float(depth[valid].max())
        depth01[valid] = (depth[valid] - dmin) / max(dmax - dmin, 1e-8)
    return depth01, mask


# ---------- 主流程 ----------

def build_for_split(root: str, split: str, save_vis: bool = False) -> None:
    """
    root: dataset/DAD-3DHeadsDataset
    split: 'train' or 'val'
    會在 {root}/{split}/depth_map 與 regionmap 產生對應影像，解析度等於原圖。
    """
    split_dir = os.path.join(root, split)
    json_path = os.path.join(split_dir, f"{split}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")

    os.makedirs(os.path.join(split_dir, "depth_map"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "regionmap"), exist_ok=True)
    if save_vis:
        os.makedirs(os.path.join(split_dir, "depth_vis"), exist_ok=True)

    faces_all = load_flame_faces_zero_based()
    face_vs = load_face_vertex_set()
    faces_face = select_face_triangles(faces_all, face_vs)

    with open(json_path) as f:
        anno_list = json.load(f)

    for idx, item in enumerate(anno_list):
        img_path = os.path.join(root, item["img_path"])
        mesh_path = os.path.join(root, item["annotation_path"])

        # 載圖與 mesh
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if img is None:
            print(f"[WARN] skip unreadable image: {img_path}")
            continue
        H, W = img.shape[:2]

        with open(mesh_path) as jf:
            m = json.load(jf)
        verts = np.array(m["vertices"], dtype=np.float32)                   # [5023,3]
        M = np.array(m["model_view_matrix"], dtype=np.float32)              # [4,4]
        P = np.array(m["projection_matrix"], dtype=np.float32)              # [4,4]
        Vh = np.concatenate([verts, np.ones((verts.shape[0], 1), np.float32)], axis=1)
        Vh_world = (M @ Vh.T).T

        # 投影 + 光柵化（全圖）
        xy, z, valid_w = project_all_vertices_full(Vh_world, P, H)
        depth01, region = rasterize_depth01_and_mask(xy, z, faces_face, W, H, valid_w)

        # 儲存（與原圖同名）
        base = os.path.splitext(os.path.basename(img_path))[0]
        depth_png = os.path.join(split_dir, "depth_map", f"{base}.png")
        region_png = os.path.join(split_dir, "regionmap", f"{base}.png")

        # depth：16-bit 單通道
        depth_u16 = (np.clip(depth01, 0.0, 1.0) * 65535.0).astype(np.uint16)
        # 背景（region==0）就留 0
        depth_u16[region == 0] = 0
        cv2.imwrite(depth_png, depth_u16, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        # region：8-bit 0/255
        region_u8 = (region * 255).astype(np.uint8)
        cv2.imwrite(region_png, region_u8, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        # 可選：彩色可視化（方便肉眼檢查）
        if save_vis:
            d8 = (np.clip(depth01, 0.0, 1.0) * 255).astype(np.uint8)
            dvis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
            dvis[region == 0] = 0
            vis_png = os.path.join(split_dir, "depth_vis", f"{base}.png")
            cv2.imwrite(vis_png, dvis, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        if (idx + 1) % 50 == 0:
            print(f"[{split}] processed {idx + 1}/{len(anno_list)}")

    print(f"[{split}] done. depth_map & regionmap saved under {split_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="e.g. dataset/DAD-3DHeadsDataset")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--save-vis", action="store_true", help="also write colorized depth under depth_vis/")
    args = ap.parse_args()

    for sp in args.splits:
        build_for_split(args.dataset_root, sp, save_vis=args.save_vis)


if __name__ == "__main__":
    main()

# python model_training/data/build_face_maps.py --dataset-root dataset/DAD-3DHeadsDataset --splits train val --save-vis