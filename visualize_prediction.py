import os
import sys
import json
import yaml
from typing import Dict, Any, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

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
)
from model_training.model.utils import unravel_index, normalize_to_cube
from model_training.train.utils import any2device
from model_training.metrics.iou import SoftIoUMetric
from model_training.metrics.keypoints import FailureRate, KeypointsNME


def parse_checkpoint_path_from_argv():
    """Parse checkpoint_path from sys.argv before Hydra processes it."""
    checkpoint_path = None
    args_to_remove = []
    
    for i, arg in enumerate(sys.argv):
        if arg == "--checkpoint_path" and i + 1 < len(sys.argv):
            checkpoint_path = sys.argv[i + 1]
            args_to_remove.append(i)
            args_to_remove.append(i + 1)
            break
        elif arg.startswith("--checkpoint_path="):
            checkpoint_path = arg.split("=", 1)[1]
            args_to_remove.append(i)
            break
        elif arg.startswith("checkpoint_path="):
            checkpoint_path = arg.split("=", 1)[1]
            args_to_remove.append(i)
            break
    
    for i in sorted(args_to_remove, reverse=True):
        sys.argv.pop(i)
    
    return checkpoint_path


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


def draw_points(image_bgr: np.ndarray, pts_xy: np.ndarray, color=(0, 0, 255), radius: int = None) -> np.ndarray:
    """Draw points on image."""
    out = image_bgr.copy()
    H, W = out.shape[:2]
    rr = radius if radius is not None else max(1, int(min(H, W) * 0.005))
    for (x, y) in pts_xy.astype(int):
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(out, (int(x), int(y)), rr, color, -1, lineType=cv2.LINE_AA)
    return out


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


def visualize_prediction(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    dataset_mode: str = "val",
    max_items: Optional[int] = None,
    device: str = "cuda:0",
    save_images: bool = True,
    batch_size: int = 1
):
    """
    Visualize model predictions alongside ground truth and compute metrics.
    
    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for visualizations
        dataset_mode: "train", "val", or "test"
        max_items: Maximum number of items to visualize (None = all)
        device: Device to run inference on
        save_images: Whether to save visualization images
        batch_size: Batch size for inference
    """
    # Load config
    with open(config_path, "r") as f:
        config_all = yaml.safe_load(f)
    
    if dataset_mode not in config_all:
        raise ValueError(f"Dataset mode '{dataset_mode}' not found in config")
    
    dataset_cfg = config_all[dataset_mode]
    dataset = FlameDataset.from_config(config=dataset_cfg)
    
    # Load model
    from model_training.utils import create_logger
    from model_training.model import load_model
    from model_training.train.flame_lightning_model import FlameLightningModel
    
    logger = create_logger(__name__)
    
    # Create dummy train dataset (required by FlameLightningModel)
    train_dataset = FlameDataset.from_config(config=config_all["train"])

    # Load model
    config_all["model"]["flame_indices_config"] =  config_all["train"]["flame_indices"]
    model = load_model(config_all["model"], config_all["constants"])
    
    # Create Lightning model and load checkpoint
    config_all["load_weights"] = True
    config_all["weights_path"] = checkpoint_path
    config_all["wandb"] = {"enable": False}
    
    dad3d_net = FlameLightningModel(model=model, config=config_all, train=train_dataset, val=dataset)
    dad3d_net.eval()
    dad3d_net = dad3d_net.to(device)
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    norm_name = dataset_cfg.get("transform", {}).get("normalize", "imagenet")
    img_size = dataset.img_size
    stride = dataset_cfg.get("stride", 4)
    
    # ========== Initialize Metrics (same as FlameLightningModel) ==========
    # Heatmap IoU metric
    heatmap_iou_metric = SoftIoUMetric(compute_on_step=False).to(device)
    
    # Region IoU metric (for face region mask)
    region_iou_metric = SoftIoUMetric(compute_on_step=False).to(device)
    
    metrics_2d = MetricCollection({
        "fr_2d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
        "fr_2d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
        "nme_2d": KeypointsNME(compute_on_step=False),
    }).to(device)
    
    metrics_reprojection = MetricCollection({
        "reproject_fr_2d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
        "reproject_fr_2d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
        "reproject_nme_2d": KeypointsNME(compute_on_step=False),
    }).to(device)
    
    metrics_3d = MetricCollection({
        "fr_3d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
        "fr_3d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
        "nme_3d": KeypointsNME(compute_on_step=False),
    }).to(device)
    
    # Load FLAME indices
    flame_indices = {}
    for key, value in config_all["train"]["flame_indices"]["files"].items():
        flame_indices[key] = np.load(os.path.join(config_all["train"]["flame_indices"]["folder"], value))
    
    # ========== Initialize Loss Accumulators ==========
    loss_accumulators = {}
    loss_counts = {}
    
    # Helper to get 2D keypoints from outputs
    def get_keypoints_2d(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if OUTPUT_2D_LANDMARKS in outputs.keys():
            return outputs[OUTPUT_2D_LANDMARKS] * img_size
        return float(stride) * unravel_index(outputs[OUTPUT_LANDMARKS_HEATMAP]).flip(-1)
    
    # Process each item
    num_items = len(dataset) if max_items is None else min(max_items, len(dataset))
    
    for idx in tqdm(range(num_items), desc="Processing"):
        item = dataset[idx]
        ann = dataset.data[idx]
        
        # Prepare input
        input_image = item[INPUT_IMAGE_KEY].unsqueeze(0).to(device)  # [1, C, H, W]
        
        # Prepare targets as tensors
        targets = {}
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                targets[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            elif isinstance(value, torch.Tensor):
                targets[key] = value.unsqueeze(0).to(device)
            elif isinstance(value, (list, tuple)) and key == INPUT_BBOX_KEY:
                # Convert bbox tuple/list to tensor with batch dimension
                targets[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                targets[key] = value
        
        # Run inference
        with torch.no_grad():
            output = dad3d_net.model(input_image)
        
        # ========== Compute Losses (same as test.py) ==========
        with torch.no_grad():
            total_loss, loss_dict = dad3d_net.criterion(output, targets, epoch=999)
            
            # Accumulate losses
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in loss_accumulators:
                    loss_accumulators[loss_name] = 0.0
                    loss_counts[loss_name] = 0
                loss_accumulators[loss_name] += loss_value.item()
                loss_counts[loss_name] += 1
            
            # Also accumulate total loss
            if "total_loss" not in loss_accumulators:
                loss_accumulators["total_loss"] = 0.0
                loss_counts["total_loss"] = 0
            loss_accumulators["total_loss"] += total_loss.item()
            loss_counts["total_loss"] += 1
        
        # ========== Compute Metrics ==========
        # Heatmap IoU metric
        if OUTPUT_LANDMARKS_HEATMAP in output and TARGET_LANDMARKS_HEATMAP in targets:
            heatmap_iou_metric(
                output[OUTPUT_LANDMARKS_HEATMAP].sigmoid(),
                targets[TARGET_LANDMARKS_HEATMAP]
            )
        
        # Region IoU metric
        if OUTPUT_REGION in output and TARGET_FACE_REGION in targets:
            region_iou_metric(
                output[OUTPUT_REGION].sigmoid(),
                targets[TARGET_FACE_REGION]
            )
        
        # 2D Landmarks metrics
        process_2d_branch = OUTPUT_2D_LANDMARKS in output or OUTPUT_LANDMARKS_HEATMAP in output
        if process_2d_branch and TARGET_2D_LANDMARKS in targets:
            presence = targets[TARGET_2D_LANDMARKS_PRESENCE]
            outputs_2d = get_keypoints_2d(output) * presence[..., None]
            targets_2d = targets[TARGET_2D_LANDMARKS] * presence[..., None] * img_size
            metrics_2d(outputs_2d, {"keypoints": targets_2d, "bboxes": targets[INPUT_BBOX_KEY]})
        
        # Reprojection metrics (3D mesh projected to 2D) - use pre-computed vertices
        if OUTPUT_2D_VERTICES in output and TARGET_2D_FULL_LANDMARKS in targets:
            reprojected_pred = output[OUTPUT_2D_VERTICES][:, flame_indices["face"]]
            reprojected_gt = targets[TARGET_2D_FULL_LANDMARKS][:, flame_indices["face"]]
            metrics_reprojection(
                reprojected_pred,
                {"keypoints": reprojected_gt, "bboxes": targets[INPUT_BBOX_KEY]}
            )
        
        # 3D vertices metrics - use pre-computed vertices
        if OUTPUT_3D_VERTICES in output and TARGET_3D_MODEL_VERTICES in targets:
            pred_3d_vertices = output[OUTPUT_3D_VERTICES]
            metrics_3d(
                normalize_to_cube(pred_3d_vertices[:, flame_indices["face"]]),
                {"keypoints": normalize_to_cube(targets[TARGET_3D_MODEL_VERTICES][:, flame_indices["face"]])}
            )
        
        # ========== Save Visualizations (if enabled) ==========
        if save_images:
            item_dir = os.path.join(output_dir, str(idx))
            os.makedirs(item_dir, exist_ok=True)
            
            # Convert to numpy for visualization
            img_bgr = tensor_to_bgr_uint8(item[INPUT_IMAGE_KEY], normalize_name=norm_name)
            H_vis, W_vis = img_bgr.shape[:2]
            
            # ========== Ground Truth Visualizations ==========
            # GT Landmarks
            lm_gt_norm = item[TARGET_2D_LANDMARKS]
            lm_gt_pix = (lm_gt_norm * img_size).astype(np.float32)
            if TARGET_2D_LANDMARKS_PRESENCE in item:
                presence_np = item[TARGET_2D_LANDMARKS_PRESENCE].astype(bool)
                lm_gt_pix_vis = lm_gt_pix[presence_np]
            else:
                lm_gt_pix_vis = lm_gt_pix
            img_gt_kp = draw_points(img_bgr, lm_gt_pix_vis, color=(0, 0, 255))
            
            # GT Heatmap
            if TARGET_LANDMARKS_HEATMAP in item:
                hm_gt = item[TARGET_LANDMARKS_HEATMAP]
                img_gt_hm = overlay_heatmap_on_image(img_bgr, hm_gt, alpha=0.5)
            else:
                img_gt_hm = img_bgr.copy()
            
            # GT Full landmarks (projected vertices) - used for reproject_nme_2d metric
            if TARGET_2D_FULL_LANDMARKS in item:
                verts2d_gt = item[TARGET_2D_FULL_LANDMARKS].astype(np.float32)
                # Use face subset if available (same as metrics)
                if "face" in flame_indices:
                    verts2d_gt_face = verts2d_gt[flame_indices["face"]]
                else:
                    verts2d_gt_face = verts2d_gt
                img_gt_proj = draw_points(img_bgr, verts2d_gt_face, color=(0, 255, 0), radius=1)
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
            lm_pred = extract_landmarks_from_output(output, img_size, stride)
            if len(lm_pred) > 0:
                img_pred_kp = draw_points(img_bgr, lm_pred, color=(255, 0, 0))
            else:
                img_pred_kp = img_bgr.copy()
            
            # Pred Heatmap
            if OUTPUT_LANDMARKS_HEATMAP in output:
                hm_pred = torch.sigmoid(output[OUTPUT_LANDMARKS_HEATMAP]).detach().cpu().numpy()[0]  # [N, H, W]
                img_pred_hm = overlay_heatmap_on_image(img_bgr, hm_pred, alpha=0.5)
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
                if "face" in flame_indices:
                    verts2d_pred_face = verts2d_pred[flame_indices["face"]]
                else:
                    verts2d_pred_face = verts2d_pred
                img_pred_proj = draw_points(img_bgr, verts2d_pred_face.astype(np.float32), color=(255, 0, 0), radius=1)
            else:
                img_pred_proj = img_bgr.copy()
            
            # ========== Create Comparison Images ==========
            comp_landmarks = create_comparison_image(img_gt_kp, img_pred_kp, "GT Landmarks", "Pred Landmarks")
            comp_heatmap = create_comparison_image(img_gt_hm, img_pred_hm, "GT Heatmap", "Pred Heatmap")
            comp_depth = create_comparison_image(img_gt_depth_overlay, img_pred_depth_overlay, "GT Depth", "Pred Depth")
            comp_region = create_comparison_image(img_gt_region_overlay, img_pred_region_overlay, "GT Region", "Pred Region")
            comp_projected = create_comparison_image(img_gt_proj, img_pred_proj, "GT Reprojected", "Pred Reprojected")
            
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
            orig = cv2.imread(os.path.join(dataset.config["dataset_root"], ann["img_path"]))
            if orig is not None:
                cv2.imwrite(os.path.join(item_dir, "original.png"), orig)
    
    # ========== Compute and Print Final Metrics ==========
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    
    all_metrics = {}
    
    # ========== Losses (same format as test.py) ==========
    logger.info("\nLosses:")
    logger.info("-" * 40)
    for loss_name in sorted(loss_accumulators.keys()):
        avg_loss = loss_accumulators[loss_name] / loss_counts[loss_name]
        all_metrics[f"loss/{loss_name}"] = avg_loss
        logger.info(f"  {loss_name}: {avg_loss:.6f}")
    
    # ========== Metrics ==========
    logger.info("\nMetrics:")
    logger.info("-" * 40)
    
    # Heatmap IoU metric
    try:
        heatmap_iou = heatmap_iou_metric.compute().item()
        all_metrics["metrics/heatmap_iou"] = heatmap_iou
        logger.info(f"  heatmap_iou: {heatmap_iou:.6f}")
    except Exception as e:
        logger.warning(f"Could not compute heatmap IoU: {e}")
    
    # Region IoU metric
    try:
        region_iou = region_iou_metric.compute().item()
        all_metrics["metrics/region_iou"] = region_iou
        logger.info(f"  region_iou: {region_iou:.6f}")
    except Exception as e:
        logger.warning(f"Could not compute region IoU: {e}")
    
    # 2D Landmarks metrics
    try:
        metrics_2d_result = metrics_2d.compute()
        for name, value in metrics_2d_result.items():
            val = value.item()
            all_metrics[f"metrics/{name}"] = val
            logger.info(f"  {name}: {val:.6f}")
    except Exception as e:
        logger.warning(f"Could not compute 2D metrics: {e}")
    
    # Reprojection metrics
    try:
        metrics_reproj_result = metrics_reprojection.compute()
        for name, value in metrics_reproj_result.items():
            val = value.item()
            all_metrics[f"metrics/{name}"] = val
            logger.info(f"  {name}: {val:.6f}")
    except Exception as e:
        logger.warning(f"Could not compute reprojection metrics: {e}")
    
    # 3D metrics
    try:
        metrics_3d_result = metrics_3d.compute()
        for name, value in metrics_3d_result.items():
            val = value.item()
            all_metrics[f"metrics/{name}"] = val
            logger.info(f"  {name}: {val:.6f}")
    except Exception as e:
        logger.warning(f"Could not compute 3D metrics: {e}")
    
    logger.info("=" * 80)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Also save a summary text file (similar to test.py output format)
    summary_file = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {dataset_mode}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Num samples: {num_items}\n")
        f.write("=" * 80 + "\n\n")
        
        # Format like test.py output
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<45} {'Value':>20}\n")
        f.write("-" * 80 + "\n")
        for name, value in sorted(all_metrics.items()):
            f.write(f"{name:<45} {value:>20.6f}\n")
        f.write("-" * 80 + "\n")
    logger.info(f"Summary saved to {summary_file}")
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model predictions and compute metrics")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="visualize/predictions", help="Output directory")
    parser.add_argument("--dataset_mode", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset mode to use")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items (None = all)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("--no_save_images", action="store_true", help="Skip saving visualization images (metrics only)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    
    args = parser.parse_args()
    
    visualize_prediction(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        dataset_mode=args.dataset_mode,
        max_items=args.max_items,
        device=args.device,
        save_images=not args.no_save_images,
        batch_size=args.batch_size
    )

# Examples:
# Run on whole dataset with metrics and visualizations:
# python visualize_prediction.py --config /home/cytseng/git/DAD-3DHeads/experiments/train/2026-01-13-00-08-06/experiment_config.yaml --checkpoint_path  /home/cytseng/git/DAD-3DHeads/experiments/train/2026-01-13-00-08-06/train_on_all_fusion_deformation_head2_3d_loss/checkpoints/epoch_0088-valid_metrics_reproject_nme_2d_1.6756.ckpt --output_dir visualize/predictions --dataset_mode val

# Run metrics only (faster, no images):
# python visualize_prediction.py --config experiments/train/2025-12-21-14-35-05/experiment_config.yaml --checkpoint_path experiments/train/2025-12-21-14-35-05/train_on_all_fusion_on_all/checkpoints/epoch_0106-valid_metrics_reproject_nme_2d_1.6748.ckpt --output_dir visualize/predictions --dataset_mode val --no_save_images
