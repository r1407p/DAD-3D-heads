from typing import Dict, Any
import numpy as np
import cv2
import torch
import json
import os
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
    TARGET_3D_MODEL_VERTICES,
    TARGET_FACE_REGION,
    TARGET_FACE_DEPTH,
    OUTPUT_2D_LANDMARKS,
    OUTPUT_LANDMARKS_HEATMAP,
    OUTPUT_3DMM_PARAMS,
    OUTPUT_3D_VERTICES,
    OUTPUT_2D_VERTICES,
    OUTPUT_DEPTH,
    OUTPUT_REGION,
    OUTPUT_RESIDUAL_DEFORMATION,
    OUTPUT_3D_VERTICES_REFINED,
    OUTPUT_2D_VERTICES_REFINED,
)


from model_training.model.utils import unravel_index

class Visualizer:
    def __init__(self, output_dir: str, dataset: FlameDataset, flame_indices: Dict[str, np.ndarray], norm_name: str, img_size: int, stride: int):
        self.output_dir = output_dir
        self.dataset = dataset
        self.flame_indices = flame_indices
        self.norm_name = norm_name
        self.img_size = img_size
        self.stride = stride

    @staticmethod
    def tensor_to_bgr_uint8(img_tensor: torch.Tensor, normalize_name: str = "imagenet") -> np.ndarray:
        """Convert tensor image to BGR uint8 numpy array."""
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

    @staticmethod
    def draw_points(image_bgr: np.ndarray, pts_xy: np.ndarray, color=(0, 0, 255), radius: int = None) -> np.ndarray:
        """Draw points on image."""
        out = image_bgr.copy()
        H, W = out.shape[:2]
        rr = radius if radius is not None else max(1, int(min(H, W) * 0.005))
        for (x, y) in pts_xy.astype(int):
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(out, (int(x), int(y)), rr, color, -1, lineType=cv2.LINE_AA)
        return out

    @staticmethod
    def draw_points_with_visibility(
        image_bgr: np.ndarray, 
        pts_xy: np.ndarray, 
        visible_mask: np.ndarray,
        color_visible=(0, 255, 0),      # Green for visible
        color_occluded=(0, 0, 255),     # Red for occluded
        radius: int = None
    ) -> np.ndarray:
        """Draw points on image with different colors for visible/occluded vertices."""
        out = image_bgr.copy()
        H, W = out.shape[:2]
        rr = radius if radius is not None else max(1, int(min(H, W) * 0.005))
        
        pts_int = pts_xy.astype(int)
        for i, (x, y) in enumerate(pts_int):
            if 0 <= x < W and 0 <= y < H:
                color = color_visible if visible_mask[i] else color_occluded
                cv2.circle(out, (int(x), int(y)), rr, color, -1, lineType=cv2.LINE_AA)
        return out

    @staticmethod
    def overlay_heatmap_on_image(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on image."""
        hm = heatmap.max(axis=0) if heatmap.ndim == 3 else heatmap
        hm = hm.astype(np.float32)
        hm -= hm.min()
        hm = hm / (hm.max() + 1e-8)
        hm_u8 = (hm * 255.0).astype(np.uint8)
        H, W = image_bgr.shape[:2]
        hm_u8 = cv2.resize(hm_u8, (W, H), interpolation=cv2.INTER_LINEAR)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        return cv2.addWeighted(image_bgr, 1.0, hm_color, alpha, 0)

    @staticmethod
    def create_comparison_image(gt_img: np.ndarray, pred_img: np.ndarray, title_gt: str = "Ground Truth", 
                            title_pred: str = "Prediction", gap: int = 10) -> np.ndarray:
        """Create side-by-side comparison image."""
        H, W = gt_img.shape[:2]
        assert gt_img.shape == pred_img.shape, "Images must have same shape"
        
        # Create combined image
        combined = np.zeros((H, W * 2 + gap, 3), dtype=np.uint8)
        combined[:, :W] = gt_img
        combined[:, W + gap:] = pred_img
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        
        (text_width_gt, text_height_gt), _ = cv2.getTextSize(title_gt, font, font_scale, thickness)
        (text_width_pred, text_height_pred), _ = cv2.getTextSize(title_pred, font, font_scale, thickness)
        
        # Draw background for text
        cv2.rectangle(combined, (5, 5), (text_width_gt + 10, text_height_gt + 15), (0, 0, 0), -1)
        cv2.rectangle(combined, (W + gap + 5, 5), (W + gap + text_width_pred + 10, text_height_pred + 15), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(combined, title_gt, (10, text_height_gt + 10), font, font_scale, color, thickness)
        cv2.putText(combined, title_pred, (W + gap + 10, text_height_pred + 10), font, font_scale, color, thickness)
        
        return combined

    @staticmethod
    def extract_landmarks_from_output(output: Dict[str, torch.Tensor], img_size: int, stride: int = 4) -> np.ndarray:
        """Extract landmark coordinates from model output."""
        if OUTPUT_2D_LANDMARKS in output:
            landmarks = output[OUTPUT_2D_LANDMARKS].detach().cpu().numpy()[0]  # [N, 2]
            landmarks = landmarks * img_size  # Denormalize
        elif OUTPUT_LANDMARKS_HEATMAP in output:
            pred_heatmap = output[OUTPUT_LANDMARKS_HEATMAP]
            # Extract landmarks from heatmap
            landmarks = unravel_index(torch.sigmoid(pred_heatmap).detach()).flip(-1)[0].cpu().numpy()  # [N, 2]
            landmarks = landmarks * stride  # Scale by stride
        else:
            return np.array([])
        return landmarks.astype(np.float32)

    def visualize_prediction(self, idx: int, item: Dict[str, Any], ann: Dict[str, Any], output: Dict[str, torch.Tensor], visible_mask: np.ndarray = None):
       
        item_dir = os.path.join(self.output_dir, str(idx))
        os.makedirs(item_dir, exist_ok=True)
        
        # Convert to numpy for visualization
        img_bgr = Visualizer.tensor_to_bgr_uint8(item[INPUT_IMAGE_KEY], normalize_name=self.norm_name)
        H_vis, W_vis = img_bgr.shape[:2]
        
        # ========== Ground Truth Visualizations ==========
        # GT Landmarks
        lm_gt_norm = item[TARGET_2D_LANDMARKS]
        lm_gt_pix = (lm_gt_norm * self.img_size).astype(np.float32)
        if TARGET_2D_LANDMARKS_PRESENCE in item:
            presence_np = item[TARGET_2D_LANDMARKS_PRESENCE].astype(bool)
            lm_gt_pix_vis = lm_gt_pix[presence_np]
        else:
            lm_gt_pix_vis = lm_gt_pix
        img_gt_kp = Visualizer.draw_points(img_bgr, lm_gt_pix_vis, color=(0, 0, 255))
        
        # GT Heatmap
        if TARGET_LANDMARKS_HEATMAP in item:
            hm_gt = item[TARGET_LANDMARKS_HEATMAP]
            img_gt_hm = Visualizer.overlay_heatmap_on_image(img_bgr, hm_gt, alpha=0.5)
        else:
            img_gt_hm = img_bgr.copy()
        
        # GT Full landmarks (projected vertices) - used for reproject_nme_2d metric
        if TARGET_2D_FULL_LANDMARKS in item:
            verts2d_gt = item[TARGET_2D_FULL_LANDMARKS].astype(np.float32)
            # Use face subset if available (same as metrics)
            if "face" in self.flame_indices:
                verts2d_gt_face = verts2d_gt[self.flame_indices["face"]]
            else:
                verts2d_gt_face = verts2d_gt
            img_gt_proj = Visualizer.draw_points(img_bgr, verts2d_gt_face, color=(0, 255, 0), radius=1)
        else:
            img_gt_proj = img_bgr.copy()
        
        # GT Depth
        if TARGET_FACE_DEPTH in item:
            depth_gt = item[TARGET_FACE_DEPTH]
            if isinstance(depth_gt, torch.Tensor):
                depth_gt = depth_gt.detach().cpu().numpy()
            depth_gt = depth_gt.squeeze().astype(np.float32)
            depth_gt_u8 = (np.clip(depth_gt, 0, 1) * 255.0).astype(np.uint8)
            depth_gt_u8 = cv2.resize(depth_gt_u8, (W_vis, H_vis), interpolation=cv2.INTER_LINEAR)
            depth_gt_color = cv2.applyColorMap(depth_gt_u8, cv2.COLORMAP_JET)
            img_gt_depth_overlay = cv2.addWeighted(img_bgr, 1.0, depth_gt_color, 0.6, 0)
        else:
            depth_gt_color = np.zeros_like(img_bgr)
            img_gt_depth_overlay = img_bgr.copy()
        
        # GT Region
        if TARGET_FACE_REGION in item:
            region_gt = item[TARGET_FACE_REGION]
            if isinstance(region_gt, torch.Tensor):
                region_gt = region_gt.detach().cpu().numpy()
            region_gt = region_gt.squeeze().astype(np.float32)
            region_gt_u8 = (np.clip(region_gt, 0, 1) * 255.0).astype(np.uint8)
            region_gt_u8 = cv2.resize(region_gt_u8, (W_vis, H_vis), interpolation=cv2.INTER_NEAREST)
            region_gt_color = cv2.applyColorMap(region_gt_u8, cv2.COLORMAP_BONE)
            img_gt_region_overlay = cv2.addWeighted(img_bgr, 1.0, region_gt_color, 0.5, 0)
        else:
            region_gt_color = np.zeros_like(img_bgr)
            img_gt_region_overlay = img_bgr.copy()
        
        # ========== Prediction Visualizations ==========
        # Pred Landmarks
        lm_pred = Visualizer.extract_landmarks_from_output(output, self.img_size, self.stride)
        if len(lm_pred) > 0:
            img_pred_kp = Visualizer.draw_points(img_bgr, lm_pred, color=(255, 0, 0))
        else:
            img_pred_kp = img_bgr.copy()
        
        # Pred Heatmap
        if OUTPUT_LANDMARKS_HEATMAP in output:
            hm_pred = torch.sigmoid(output[OUTPUT_LANDMARKS_HEATMAP]).detach().cpu().numpy()[0]  # [N, H, W]
            img_pred_hm = Visualizer.overlay_heatmap_on_image(img_bgr, hm_pred, alpha=0.5)
        else:
            img_pred_hm = img_bgr.copy()
        
        # Pred Depth
        if OUTPUT_DEPTH in output:
            depth_pred = torch.sigmoid(output[OUTPUT_DEPTH]).detach().cpu().numpy()[0, 0]  # [H, W]
            depth_pred = np.clip(depth_pred, 0, 1)
            depth_pred_u8 = (depth_pred * 255.0).astype(np.uint8)
            depth_pred_u8 = cv2.resize(depth_pred_u8, (W_vis, H_vis), interpolation=cv2.INTER_LINEAR)
            depth_pred_color = cv2.applyColorMap(depth_pred_u8, cv2.COLORMAP_JET)
            img_pred_depth_overlay = cv2.addWeighted(img_bgr, 1.0, depth_pred_color, 0.6, 0)
        else:
            depth_pred_color = np.zeros_like(img_bgr)
            img_pred_depth_overlay = img_bgr.copy()
        
        # Pred Region
        if OUTPUT_REGION in output:
            region_pred = output[OUTPUT_REGION].detach().cpu().numpy()[0, 0]  # [H, W]
            region_pred = np.clip(region_pred, 0, 1)
            region_pred_u8 = (region_pred * 255.0).astype(np.uint8)
            region_pred_u8 = cv2.resize(region_pred_u8, (W_vis, H_vis), interpolation=cv2.INTER_NEAREST)
            region_pred_color = cv2.applyColorMap(region_pred_u8, cv2.COLORMAP_BONE)
            img_pred_region_overlay = cv2.addWeighted(img_bgr, 1.0, region_pred_color, 0.5, 0)
        else:
            region_pred_color = np.zeros_like(img_bgr)
            img_pred_region_overlay = img_bgr.copy()
        
        # Pred Reprojected Vertices (OUTPUT_2D_VERTICES) - used for reproject_nme_2d metric
        if OUTPUT_2D_VERTICES in output:
            verts2d_pred = output[OUTPUT_2D_VERTICES].detach().cpu().numpy()[0]  # [N, 2]
            # Use face subset if available (same as metrics)
            if "face" in self.flame_indices:
                face_indices = self.flame_indices["face"]
                verts2d_pred_face = verts2d_pred[face_indices]
                # Also subset the visibility mask if provided
                if visible_mask is not None:
                    visible_mask_face = visible_mask[face_indices]
                else:
                    visible_mask_face = None
            else:
                verts2d_pred_face = verts2d_pred
                visible_mask_face = visible_mask
            
            # Draw with visibility coloring if mask is provided
            if visible_mask_face is not None:
                img_pred_proj = Visualizer.draw_points_with_visibility(
                    img_bgr, 
                    verts2d_pred_face.astype(np.float32), 
                    visible_mask_face,
                    color_visible=(0, 255, 0),    # Green for visible
                    color_occluded=(0, 0, 255),   # Red for occluded
                    radius=1
                )
            else:
                img_pred_proj = Visualizer.draw_points(img_bgr, verts2d_pred_face.astype(np.float32), color=(255, 0, 0), radius=1)
        else:
            img_pred_proj = img_bgr.copy()
        
        # ========== Create Comparison Images ==========
        comp_landmarks = Visualizer.create_comparison_image(img_gt_kp, img_pred_kp, "GT Landmarks", "Pred Landmarks")
        comp_heatmap = Visualizer.create_comparison_image(img_gt_hm, img_pred_hm, "GT Heatmap", "Pred Heatmap")
        comp_depth = Visualizer.create_comparison_image(img_gt_depth_overlay, img_pred_depth_overlay, "GT Depth", "Pred Depth")
        comp_region = Visualizer.create_comparison_image(img_gt_region_overlay, img_pred_region_overlay, "GT Region", "Pred Region")
        comp_projected = Visualizer.create_comparison_image(img_gt_proj, img_pred_proj, "GT Reprojected", "Pred Reprojected")
        
        # ========== Save Images ==========
        cv2.imwrite(os.path.join(item_dir, "input.png"), img_bgr)
        cv2.imwrite(os.path.join(item_dir, "comparison_landmarks.png"), comp_landmarks)
        cv2.imwrite(os.path.join(item_dir, "comparison_heatmap.png"), comp_heatmap)
        cv2.imwrite(os.path.join(item_dir, "comparison_depth.png"), comp_depth)
        cv2.imwrite(os.path.join(item_dir, "comparison_region.png"), comp_region)
        cv2.imwrite(os.path.join(item_dir, "comparison_projected.png"), comp_projected)
        
        # Save individual images
        cv2.imwrite(os.path.join(item_dir, "gt_landmarks.png"), img_gt_kp)
        cv2.imwrite(os.path.join(item_dir, "pred_landmarks.png"), img_pred_kp)
        cv2.imwrite(os.path.join(item_dir, "gt_heatmap.png"), img_gt_hm)
        cv2.imwrite(os.path.join(item_dir, "pred_heatmap.png"), img_pred_hm)
        cv2.imwrite(os.path.join(item_dir, "gt_depth.png"), depth_gt_color)
        cv2.imwrite(os.path.join(item_dir, "pred_depth.png"), depth_pred_color)
        cv2.imwrite(os.path.join(item_dir, "gt_region.png"), region_gt_color)
        cv2.imwrite(os.path.join(item_dir, "pred_region.png"), region_pred_color)
        cv2.imwrite(os.path.join(item_dir, "gt_reprojected.png"), img_gt_proj)
        cv2.imwrite(os.path.join(item_dir, "pred_reprojected.png"), img_pred_proj)
        
        # Save metadata
        x, y, w, h = item[INPUT_BBOX_KEY]
        meta = {
            "index": int(item[SAMPLE_INDEX_KEY]),
            "img_path": ann["img_path"],
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "image_size_input": [int(H_vis), int(W_vis)],
            "num_landmarks_gt": int(lm_gt_norm.shape[0]),
            "num_landmarks_pred": int(len(lm_pred)),
        }
        with open(os.path.join(item_dir, "data.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save original image if available
        orig = cv2.imread(os.path.join(self.dataset.config["dataset_root"], ann["img_path"]))
        if orig is not None:
            cv2.imwrite(os.path.join(item_dir, "original.png"), orig)
