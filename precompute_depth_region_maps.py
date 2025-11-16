#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import cv2
import torch
from skimage.draw import polygon
from tqdm import tqdm

def load_mesh(mesh_path):
    with open(mesh_path) as f:
        d = json.load(f)
    verts = np.array(d["vertices"], dtype=np.float32)           # [N,3]
    mv = np.array(d["model_view_matrix"], dtype=np.float32)     # [4,4]
    P  = np.array(d["projection_matrix"], dtype=np.float32)     # [4,4]
    verts_h = np.concatenate([verts, np.ones((verts.shape[0],1), np.float32)], axis=1)  # [N,4]
    world_h = (mv @ verts_h.T).T  # [N,4]
    return verts, world_h, P

def project_full_image(world_h, P, height):
    v2d_h = (P @ world_h.T).T                 # [N,4]
    v2d = v2d_h[:, :2] / v2d_h[:, [3]]        # perspective division
    v2d = np.stack([v2d[:, 0], height - v2d[:, 1]], axis=1)  # y axis down
    return v2d.astype(np.float32)

def load_mesh_faces(static_dir):
    faces_path = os.path.join(static_dir, "flame_mesh_faces.pt")
    if not os.path.exists(faces_path):
        raise FileNotFoundError(f"faces file not found: {faces_path} (use --static-dir to point to the folder containing flame_mesh_faces.pt)")
    faces = torch.load(faces_path)
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    faces = faces.astype(np.int32)

    idx_dir = os.path.join(static_dir, "flame_indices")
    face_vs = None
    for nm in ["face_vertices.npy", "face_verts.npy", "face_indices.npy"]:
        p = os.path.join(idx_dir, nm)
        if os.path.exists(p):
            face_vs = np.unique(np.load(p).astype(np.int32))
            break
    if face_vs is None:
        p_edges = os.path.join(idx_dir, "face_edges.npy")
        if os.path.exists(p_edges):
            edges = np.load(p_edges).astype(np.int32)
            face_vs = np.unique(edges.reshape(-1))
    if face_vs is not None and face_vs.size > 0:
        keep = np.isin(faces, face_vs).all(axis=1)
        faces = faces[keep]
    return faces

def rasterize_full(xy, depth_per_vert, faces, W, H):
    depth = np.full((H, W), np.inf, dtype=np.float32)
    mask  = np.zeros((H, W), dtype=np.uint8)
    valid = np.isfinite(xy).all(axis=1)

    for (i, j, k) in faces:
        if not (valid[i] and valid[j] and valid[k]):
            continue
        tri = xy[[i, j, k], :]  # [[x,y], [x,y], [x,y]]
        if not np.all(np.isfinite(tri)):
            continue

        rr, cc = polygon(tri[:, 1], tri[:, 0], shape=(H, W))
        if rr.size == 0:
            continue

        ax, ay = tri[0]; bx, by = tri[1]; cx, cy = tri[2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-8:
            continue

        px = cc.astype(np.float32) + 0.5
        py = rr.astype(np.float32) + 0.5
        alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        beta  = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        gamma = 1.0 - alpha - beta

        z_tri = depth_per_vert[[i, j, k]]  # [3]
        z_interp = alpha * z_tri[0] + beta * z_tri[1] + gamma * z_tri[2]

        cur = depth[rr, cc]
        closer = z_interp < cur
        if np.any(closer):
            depth[rr[closer], cc[closer]] = z_interp[closer]
            mask[rr[closer], cc[closer]] = 1

    valid_pix = (mask > 0) & np.isfinite(depth)
    depth01 = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid_pix):
        dmin = float(depth[valid_pix].min())
        dmax = float(depth[valid_pix].max())
        denom = max(dmax - dmin, 1e-8)
        depth01[valid_pix] = (depth[valid_pix] - dmin) / denom
    return depth01, mask  # full-res

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="e.g. dataset/DAD-3DHeadsDataset")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val", "test"])
    ap.add_argument("--static-dir", required=True,
                    help="folder containing flame_mesh_faces.pt and flame_indices, e.g. model/static")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing output files if true")
    ap.add_argument("--depth-dtype", choices=["uint16", "uint8"], default="uint16",
                    help="output depth map format (uint16 is recommended to reduce quantization error)")
    args = ap.parse_args()

    faces = load_mesh_faces(args.static_dir)

    for split in args.splits:
        split_dir = os.path.join(args.dataset_root, split)
        images_dir = os.path.join(split_dir, "images")
        ann_json   = os.path.join(split_dir, f"{split}.json")
        depth_dir  = os.path.join(split_dir, "depth_map")
        region_dir = os.path.join(split_dir, "regionmap")
        ensure_dir(depth_dir)
        ensure_dir(region_dir)

        if not os.path.exists(ann_json):
            print(f"[skip] {ann_json} not found")
            continue

        with open(ann_json) as f:
            annos = json.load(f)

        pbar = tqdm(annos, desc=f"Precompute {split}")
        for item in pbar:
            img_rel = item["img_path"]
            ann_rel = item["annotation_path"]
            img_rel = img_rel.replace("DAD-3DHeadsDataset/", "")
            ann_rel = ann_rel.replace("DAD-3DHeadsDataset/", "")
            img_path = os.path.join(args.dataset_root, img_rel) if not os.path.isabs(img_rel) else img_rel
            ann_path = os.path.join(args.dataset_root, ann_rel) if not os.path.isabs(ann_rel) else ann_rel
            base = os.path.splitext(os.path.basename(img_rel))[0]
            depth_out  = os.path.join(depth_dir,  base + ".png")
            region_out = os.path.join(region_dir, base + ".png")
            if (not args.overwrite) and os.path.exists(depth_out) and os.path.exists(region_out):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                pbar.write(f"[warn] failed to read {img_path}")
                continue
            H, W = img.shape[:2]

            _, world_h, P = load_mesh(ann_path)
            xy = project_full_image(world_h, P, height=H)

            z = world_h[:, 2].astype(np.float32)
            depth_per_vert = -z if np.median(z) < 0 else z

            depth01, region = rasterize_full(xy, depth_per_vert, faces, W, H)

            if args.depth_dtype == "uint16":
                depth_u16 = np.clip(depth01 * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
                cv2.imwrite(depth_out, depth_u16)
            else:
                depth_u8 = np.clip(depth01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
                cv2.imwrite(depth_out, depth_u8)

            cv2.imwrite(region_out, (region * 255).astype(np.uint8))

if __name__ == "__main__":
    main()
