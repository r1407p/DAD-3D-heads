from .vertices_3d_loss import Vertices3DLoss
from .reprojection_loss import ReprojectionLoss
from .landmarks_loss_w_visibility import LandmarksLossWVisibility
from .keypoint_losses import IoULoss
from .depth_loss import DepthLoss
from .region_loss import RegionLoss

__all__ = ["Vertices3DLoss", "ReprojectionLoss",
           "LandmarksLossWVisibility", "IoULoss", "DepthLoss", "RegionLoss"]
