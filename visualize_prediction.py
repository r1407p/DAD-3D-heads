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
import logging
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
import numpy as np
from typing import Optional
faces = torch.load('model_training/model/static/flame_mesh_faces.pt').numpy()
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return seed

set_seed(42)
def compute_vertex_visibility(
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    camera_position: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute visibility mask for 3D vertices based on backface culling.
    
    A vertex is considered visible if at least one of its adjacent faces
    is front-facing (normal pointing towards the camera).
    
    Args:
        vertices_3d: 3D vertex positions, shape (N, 3)
        faces: Triangle face indices, shape (F, 3)
        camera_position: Camera position in 3D space. If None, assumes 
                        orthographic projection with camera looking along -Z axis
                        (camera at [0, 0, +inf])
    
    Returns:
        visibility: Boolean array of shape (N,), True if vertex is visible
    """
    vertices_3d = np.asarray(vertices_3d, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    
    num_vertices = vertices_3d.shape[0]
    num_faces = faces.shape[0]
    
    # Get vertices for each face
    v0 = vertices_3d[faces[:, 0]]  # (F, 3)
    v1 = vertices_3d[faces[:, 1]]  # (F, 3)
    v2 = vertices_3d[faces[:, 2]]  # (F, 3)
    
    # Compute face normals using cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)  # (F, 3)
    
    # Normalize face normals
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    face_normals = face_normals / norms
    
    # Compute face centers
    face_centers = (v0 + v1 + v2) / 3.0  # (F, 3)
    
    # Compute view direction for each face
    if camera_position is None:
        # Orthographic projection: camera looking along -Z axis
        view_directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        view_directions = np.broadcast_to(view_directions, (num_faces, 3))
    else:
        camera_position = np.asarray(camera_position, dtype=np.float32)
        view_directions = camera_position - face_centers
        view_norms = np.linalg.norm(view_directions, axis=1, keepdims=True)
        view_norms = np.maximum(view_norms, 1e-8)
        view_directions = view_directions / view_norms
    
    # Face is visible if normal points towards camera (positive dot product)
    dot_products = np.sum(face_normals * view_directions, axis=1)  # (F,)
    face_visible = dot_products > 0  # (F,)
    
    # A vertex is visible if at least one of its adjacent faces is visible
    vertex_visible = np.zeros(num_vertices, dtype=bool)
    
    # For each visible face, mark its vertices as visible
    visible_face_indices = np.where(face_visible)[0]
    for face_idx in visible_face_indices:
        vertex_visible[faces[face_idx]] = True
    
    return vertex_visible

def compute_vertex_visibility(
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    camera_position: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute visibility mask for 3D vertices based on back-face culling.

    A vertex is considered visible if at least one of its adjacent faces
    is front-facing (normal pointing towards the camera).

    Args:
        vertices_3d: (N, 3) array of vertex positions
        faces: (F, 3) array of triangle indices
        camera_position:
            - None: assume orthographic camera looking along -Z
                    (view direction = [0, 0, -1])
            - (3,): camera position in world coordinates (perspective)

    Returns:
        visibility: (N,) boolean array
    """
    vertices_3d = np.asarray(vertices_3d)
    faces = np.asarray(faces)

    N = vertices_3d.shape[0]
    F = faces.shape[0]

    # --- Compute face normals ---
    v0 = vertices_3d[faces[:, 0]]
    v1 = vertices_3d[faces[:, 1]]
    v2 = vertices_3d[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)

    # Avoid zero-area faces
    norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid = norm.squeeze() > 1e-8
    face_normals[valid] /= norm[valid]

    # --- Compute view direction ---
    if camera_position is None:
        # Orthographic: camera looking towards -Z
        view_dirs = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        view_dirs = np.tile(view_dirs[None, :], (F, 1))
    else:
        camera_position = np.asarray(camera_position).reshape(1, 3)
        face_centers = (v0 + v1 + v2) / 3.0
        view_dirs = camera_position - face_centers
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)

    # --- Front-facing test ---
    # Face is visible if normal points toward camera
    facing = np.sum(face_normals * view_dirs, axis=1) > 0.0

    # --- Accumulate vertex visibility ---
    visibility = np.zeros(N, dtype=bool)

    visible_faces = faces[facing]
    visibility[visible_faces.reshape(-1)] = True

    return visibility


def get_keypoints_2d(outputs: Dict[str, torch.Tensor], img_size: int, stride: int) -> torch.Tensor:
    if OUTPUT_2D_LANDMARKS in outputs.keys():
        return outputs[OUTPUT_2D_LANDMARKS] * img_size
    return float(stride) * unravel_index(outputs[OUTPUT_LANDMARKS_HEATMAP]).flip(-1)

class MetricTracker:
    def __init__(self, device: str, flame_indices: Dict[str, np.ndarray], img_size: int, stride: int, logger: logging.Logger):
        self.device = device
        self.metrics = MetricTracker.create_metrics(device)
        self.loss_accum = {}
        self.loss_counts = {}
        self.flame_indices = flame_indices
        self.img_size = img_size
        self.stride = stride
        self.logger = logger

    @staticmethod
    def create_metrics(device: str) -> Dict[str, Any]:
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

    
    def update_losses(self, loss_dict, total_loss):
        for k, v in {**loss_dict, "total_loss": total_loss}.items():
            self.loss_accum[k] = self.loss_accum.get(k, 0.0) + v.item()
            self.loss_counts[k] = self.loss_counts.get(k, 0) + 1

    def compute_metrics(self, output: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        if OUTPUT_LANDMARKS_HEATMAP in output and TARGET_LANDMARKS_HEATMAP in targets:
            self.metrics["heatmap_iou"](
                output[OUTPUT_LANDMARKS_HEATMAP].sigmoid(),
                targets[TARGET_LANDMARKS_HEATMAP]
            )
        
        # Region IoU metric
        if OUTPUT_REGION in output and TARGET_FACE_REGION in targets:
            self.metrics["region_iou"](
                output[OUTPUT_REGION].sigmoid(),
                targets[TARGET_FACE_REGION]
            )
        
        # 2D Landmarks metrics
        process_2d_branch = OUTPUT_2D_LANDMARKS in output or OUTPUT_LANDMARKS_HEATMAP in output
        if process_2d_branch and TARGET_2D_LANDMARKS in targets:
            presence = targets[TARGET_2D_LANDMARKS_PRESENCE]
            outputs_2d = get_keypoints_2d(output, self.img_size, self.stride) * presence[..., None]
            targets_2d = targets[TARGET_2D_LANDMARKS] * presence[..., None] * self.img_size
            self.metrics["metrics_2d"](outputs_2d, {"keypoints": targets_2d, "bboxes": targets[INPUT_BBOX_KEY]})
        
        # Reprojection metrics (3D mesh projected to 2D) - use pre-computed vertices
        if OUTPUT_2D_VERTICES in output and TARGET_2D_FULL_LANDMARKS in targets:
            reprojected_pred = output[OUTPUT_2D_VERTICES][:, self.flame_indices["face"]]
            reprojected_gt = targets[TARGET_2D_FULL_LANDMARKS][:, self.flame_indices["face"]]
            self.metrics["metrics_reprojection"](
                reprojected_pred,
                {"keypoints": reprojected_gt, "bboxes": targets[INPUT_BBOX_KEY]}
            )
        
        # 3D vertices metrics - use pre-computed vertices
        if OUTPUT_3D_VERTICES in output and TARGET_3D_MODEL_VERTICES in targets:
            pred_3d_vertices = output[OUTPUT_3D_VERTICES]
            self.metrics["metrics_3d"](
                normalize_to_cube(pred_3d_vertices[:, self.flame_indices["face"]]),
                {"keypoints": normalize_to_cube(targets[TARGET_3D_MODEL_VERTICES][:, self.flame_indices["face"]])}
            )
        
        if OUTPUT_2D_VERTICES_REFINED in output and TARGET_2D_FULL_LANDMARKS in targets:
            reprojected_refined_pred = output[OUTPUT_2D_VERTICES_REFINED][:, self.flame_indices["face"]]
            reprojected_refined_gt = targets[TARGET_2D_FULL_LANDMARKS][:, self.flame_indices["face"]]
            self.metrics["refined_metrics_reprojection"](
                reprojected_refined_pred,
                {"keypoints": reprojected_refined_gt, "bboxes": targets[INPUT_BBOX_KEY]}
            )
        
        if OUTPUT_3D_VERTICES_REFINED in output and TARGET_3D_MODEL_VERTICES in targets:
            pred_3d_vertices_refined = output[OUTPUT_3D_VERTICES_REFINED]
            self.metrics["refined_metrics_3d"](
                normalize_to_cube(pred_3d_vertices_refined[:, self.flame_indices["face"]]),
                {"keypoints": normalize_to_cube(targets[TARGET_3D_MODEL_VERTICES][:, self.flame_indices["face"]])}
            )

    def summarize_metrics(self, output_dir: str, dataset_mode: str, checkpoint_path: str, num_items: int):
            # ========== Compute and Print Final Metrics ==========
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 80)
        
        all_metrics = {}
        
        # ========== Losses (same format as test.py) ==========
        self.logger.info("\nLosses:")
        self.logger.info("-" * 40)
        for loss_name in sorted(self.loss_accum.keys()):
            avg_loss = self.loss_accum[loss_name] / self.loss_counts[loss_name]
            all_metrics[f"loss/{loss_name}"] = avg_loss
            self.logger.info(f"  {loss_name}: {avg_loss:.6f}")
        
        # ========== Metrics ==========
        self.logger.info("\nMetrics:")
        self.logger.info("-" * 40)
        
        # Heatmap IoU metric
        try:
            heatmap_iou = self.metrics["heatmap_iou"].compute().item()
            all_metrics["metrics/heatmap_iou"] = heatmap_iou
            self.logger.info(f"  heatmap_iou: {heatmap_iou:.6f}")
        except Exception as e:
            self.logger.warning(f"Could not compute heatmap IoU: {e}")
        
        # Region IoU metric
        try:
            region_iou = self.metrics["region_iou"].compute().item()
            all_metrics["metrics/region_iou"] = region_iou
            self.logger.info(f"  region_iou: {region_iou:.6f}")
        except Exception as e:
            self.logger.warning(f"Could not compute region IoU: {e}")
        
        # 2D Landmarks metrics
        try:
            metrics_2d_result = self.metrics["metrics_2d"].compute()
            for name, value in metrics_2d_result.items():
                val = value.item()
                all_metrics[f"metrics/{name}"] = val
                self.logger.info(f"  {name}: {val:.6f}")
        except Exception as e:
            self.logger.warning(f"Could not compute 2D metrics: {e}")
        
        # Reprojection metrics
        try:
            metrics_reproj_result = self.metrics["metrics_reprojection"].compute()
            for name, value in metrics_reproj_result.items():
                val = value.item()
                all_metrics[f"metrics/{name}"] = val
                self.logger.info(f"  {name}: {val:.6f}")
        except Exception as e:
            self.logger.warning(f"Could not compute reprojection metrics: {e}")
        
        # 3D metrics
        try:
            metrics_3d_result = self.metrics["metrics_3d"].compute()
            for name, value in metrics_3d_result.items():
                val = value.item()
                all_metrics[f"metrics/{name}"] = val
                self.logger.info(f"  {name}: {val:.6f}")
        except Exception as e:
            self.logger.warning(f"Could not compute 3D metrics: {e}")

        # Refined Reprojection metrics
        try:
            refined_metrics_reproj_result = self.metrics["refined_metrics_reprojection"].compute()
            for name, value in refined_metrics_reproj_result.items():
                val = value.item()
                all_metrics[f"metrics/{name}"] = val
                self.logger.info(f"  {name}: {val:.6f}")
        except Exception as e:
            self.logger.warning(f"Could not compute refined reprojection metrics: {e}")

        # Refined 3D metrics
        try:
            refined_metrics_3d_result = self.metrics["refined_metrics_3d"].compute()
            for name, value in refined_metrics_3d_result.items():
                val = value.item()
                all_metrics[f"metrics/{name}"] = val
                self.logger.info(f"  {name}: {val:.6f}")
        except Exception as e:
            self.logger.warning(f"Could not compute refined 3D metrics: {e}")

        self.logger.info("=" * 80)
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")
        
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
        self.logger.info(f"Summary saved to {summary_file}")
        return all_metrics

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
    # Load FLAME indices
    flame_indices = {}
    for key, value in config_all["train"]["flame_indices"]["files"].items():
        flame_indices[key] = np.load(os.path.join(config_all["train"]["flame_indices"]["folder"], value))
    
    visualizer = Visualizer(output_dir, dataset, flame_indices, norm_name, img_size, stride)
    metric_tracker = MetricTracker(device, flame_indices, img_size, stride, logger)

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
            metric_tracker.update_losses(loss_dict, total_loss)
            metric_tracker.compute_metrics(output, targets)
        
        # Compute vertex visibility for visualization
        flame_params = dad3d_net.model.head_mesh.flame_params(output[OUTPUT_3DMM_PARAMS])
        rotated_vertices = dad3d_net.model.head_mesh.flame.to_rot(output[OUTPUT_3D_VERTICES], flame_params)
        visible_vertices = compute_vertex_visibility(rotated_vertices.cpu().numpy()[0], faces)

        # ========== Save Visualizations (if enabled) ==========
        if save_images:
            visualizer.visualize_prediction(idx, item, ann, output, visible_mask=visible_vertices)

    all_metrics = metric_tracker.summarize_metrics(output_dir, dataset_mode, checkpoint_path, num_items)

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
