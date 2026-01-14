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
    OUTPUT_RESIDUAL_DEFORMATION,
    OUTPUT_3D_VERTICES_REFINED,
    OUTPUT_2D_VERTICES_REFINED,
)
from model_training.model.utils import unravel_index, normalize_to_cube
from model_training.train.utils import any2device
from model_training.metrics.iou import SoftIoUMetric
from model_training.metrics.keypoints import FailureRate, KeypointsNME
from visualizer import Visualizer


def create_metrics(device: str):
    metrics = {
        "heatmap_iou": SoftIoUMetric(compute_on_step=False).to(device),
        "region_iou": SoftIoUMetric(compute_on_step=False).to(device),
        "metrics_2d": MetricCollection({
            "fr_2d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
            "fr_2d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
            "nme_2d": KeypointsNME(compute_on_step=False),
        }).to(device),
        "metrics_reprojection": MetricCollection({
            "reproject_fr_2d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
            "reproject_fr_2d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
            "reproject_nme_2d": KeypointsNME(compute_on_step=False),
        }).to(device),
        "metrics_3d": MetricCollection({
            "fr_3d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
            "fr_3d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
            "nme_3d": KeypointsNME(compute_on_step=False),
        }).to(device),
        "refined_metrics_reprojection": MetricCollection({
            "refined_reproject_fr_2d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
            "refined_reproject_fr_2d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
            "refined_reproject_nme_2d": KeypointsNME(compute_on_step=False),
        }).to(device),
        "refined_metrics_3d": MetricCollection({
            "refined_fr_3d_005": FailureRate(compute_on_step=False, threshold=0.05, below=True),
            "refined_fr_3d_01": FailureRate(compute_on_step=False, threshold=0.1, below=True),
            "refined_nme_3d": KeypointsNME(compute_on_step=False),
        }).to(device),
    }
    return metrics


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
    metrics = create_metrics(device)
    heatmap_iou_metric = metrics["heatmap_iou"]
    region_iou_metric = metrics["region_iou"]
    metrics_2d = metrics["metrics_2d"]
    metrics_reprojection = metrics["metrics_reprojection"]
    metrics_3d = metrics["metrics_3d"]
    refined_metrics_reprojection = metrics["refined_metrics_reprojection"]
    refined_metrics_3d = metrics["refined_metrics_3d"]
    
    # Load FLAME indices
    flame_indices = {}
    for key, value in config_all["train"]["flame_indices"]["files"].items():
        flame_indices[key] = np.load(os.path.join(config_all["train"]["flame_indices"]["folder"], value))
    
    visualizer = Visualizer(output_dir, dataset, flame_indices, norm_name, img_size, stride)
    
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
        
        if OUTPUT_2D_VERTICES_REFINED in output and TARGET_2D_FULL_LANDMARKS in targets:
            reprojected_refined_pred = output[OUTPUT_2D_VERTICES_REFINED][:, flame_indices["face"]]
            reprojected_refined_gt = targets[TARGET_2D_FULL_LANDMARKS][:, flame_indices["face"]]
            refined_metrics_reprojection(
                reprojected_refined_pred,
                {"keypoints": reprojected_refined_gt, "bboxes": targets[INPUT_BBOX_KEY]}
            )
        
        if OUTPUT_3D_VERTICES_REFINED in output and TARGET_3D_MODEL_VERTICES in targets:
            pred_3d_vertices_refined = output[OUTPUT_3D_VERTICES_REFINED]
            refined_metrics_3d(
                normalize_to_cube(pred_3d_vertices_refined[:, flame_indices["face"]]),
                {"keypoints": normalize_to_cube(targets[TARGET_3D_MODEL_VERTICES][:, flame_indices["face"]])}
            )
        
        # ========== Save Visualizations (if enabled) ==========
        if save_images:
            visualizer.visualize_prediction(idx, item, ann, output)
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

    # Refined Reprojection metrics
    try:
        refined_metrics_reproj_result = refined_metrics_reprojection.compute()
        for name, value in refined_metrics_reproj_result.items():
            val = value.item()
            all_metrics[f"metrics/{name}"] = val
            logger.info(f"  {name}: {val:.6f}")
    except Exception as e:
        logger.warning(f"Could not compute refined reprojection metrics: {e}")

    # Refined 3D metrics
    try:
        refined_metrics_3d_result = refined_metrics_3d.compute()
        for name, value in refined_metrics_3d_result.items():
            val = value.item()
            all_metrics[f"metrics/{name}"] = val
            logger.info(f"  {name}: {val:.6f}")
    except Exception as e:
        logger.warning(f"Could not compute refined 3D metrics: {e}")

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
    parser.add_argument("--output_dir", type=str, default="visualize/predictions_refactored", help="Output directory")
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
