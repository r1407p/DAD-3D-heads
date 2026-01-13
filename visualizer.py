
from typing import Dict
import numpy as np
import cv2
import torch

from model_training.data.config import (
    OUTPUT_2D_LANDMARKS,
    OUTPUT_LANDMARKS_HEATMAP,
)
from model_training.model.utils import unravel_index

class Visualizer:
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
