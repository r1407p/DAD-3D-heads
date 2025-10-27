import os
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import cv2
import json
from skimage.draw import polygon
import numpy as np
import torch
import cv2
import os
from model_training.model.flame import calculate_rpy, FlameParams, FLAME_CONSTS
from model_training.utils import load_indices_from_npy
from inference.uv_texture import UVTextureCreator
from inference.pncc_estimator import PNCCEstimator
from utils import get_relative_path

# region visualization
POINT_COLOR = (255, 0, 0)
EDGE_COLOR = (39, 48, 218)
OPACITY = .6
KEYPOINTS_INDICES_DIR = "model_training/model/static/face_keypoints"
FLAME_IDICES_DIR = "model_training/model/static/flame_indices/"


def draw_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Points are expected to have integer coordinates.
    """
    radius = max(1, int(min(image.shape[:2]) * 0.005))
    for pt in points:
        cv2.circle(image, (pt[0], pt[1]), radius, POINT_COLOR, -1)
    return image


def draw_landmarks(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    image = draw_points(image, predictions["points"])
    return image


def draw_3d_landmarks(predictions: Dict[str, torch.Tensor], image: np.ndarray, subset: str = "191") -> np.ndarray:
    if subset != "191" and subset != "445":
        ValueError("Invalid keypoints subset provided.\n"
                   "Available options are: 191, 445")
    subset_dir = get_relative_path(os.path.join(KEYPOINTS_INDICES_DIR, f"keypoints_{subset}"), __file__)
    projected_vertices = predictions["projected_vertices"].squeeze().numpy().astype(int)
    points = []
    for subs_file in os.listdir(subset_dir):
        subs_file_path = os.path.join(subset_dir, subs_file)
        points.extend(np.take(projected_vertices, load_indices_from_npy(subs_file_path), axis=0))
    return draw_points(image, points)


def draw_mesh(predictions: Dict[str, torch.Tensor], image: np.ndarray, subset: str = "head") -> np.ndarray:
    if subset != "face" and subset != "head":
        ValueError("Invalid FLAME mesh vertices subset provided.\n"
                   "Available options are: face, head")

    mesh_vis = image.copy()
    output = image.copy()
    projected_vertices = predictions["projected_vertices"].squeeze().numpy().astype(int)
    edges = np.load(get_relative_path(os.path.join(FLAME_IDICES_DIR, f"{subset}_edges.npy"), __file__))

    for edge in edges:
        pt1, pt2 = edge[0], edge[1]
        cv2.line(mesh_vis, projected_vertices[pt1], projected_vertices[pt2], EDGE_COLOR, 1, cv2.LINE_AA)

    cv2.addWeighted(mesh_vis, OPACITY, output, 1 - OPACITY, 0, output)
    return mesh_vis


def draw_pose(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    params_3dmm = predictions["3dmm_params"].float()
    flame_params = FlameParams.from_3dmm(params_3dmm, FLAME_CONSTS)
    rpy = calculate_rpy(flame_params)

    tdx, tdy = image.shape[1] // 2, image.shape[0] // 2

    roll = rpy.roll * np.pi / 180
    pitch = rpy.pitch * np.pi / 180
    yaw = -(rpy.yaw * np.pi / 180)

    size = image.shape[0] // 10

    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), int(image.shape[0] * 0.005))
    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), int(image.shape[0] * 0.005))
    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), int(image.shape[0] * 0.005))

    return image


def get_uv_texture(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    texture_predictor = UVTextureCreator()
    return texture_predictor(image, predictions)


def get_pncc(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    pncc_predictor = PNCCEstimator()
    return pncc_predictor(image, predictions)

# endregion

def get_mesh(predictions: Dict[str, torch.Tensor], *args: Any) -> Tuple[np.ndarray, np.ndarray]:
    vertices = predictions['3d_vertices'].numpy()
    faces = torch.load('model_training/model/static/flame_mesh_faces.pt').numpy() + 1.
    return vertices, faces


def get_flame_params(predictions: Dict[str, torch.Tensor], *args: Any) -> Dict[str, List[float]]:
    params_3dmm = predictions['3dmm_params']
    flame_params = FlameParams.from_3dmm(params_3dmm, FLAME_CONSTS)
    result = {k: v[0].tolist() for k, v in vars(flame_params).items()}
    return result


# region saving
class ImageSaver:
    def __init__(self) -> None:
        self.extension = '.png'

    def __call__(self, image: np.ndarray, output_path: str) -> None:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class MeshSaver:
    def __init__(self) -> None:
        self.extension = '.obj'

    def __call__(self, mesh: Tuple[np.ndarray, np.ndarray], output_path: str) -> None:
        """
        mesh: tuple (vertices, faces)
        Vertices: [N, 3]. Faces: [N, 3], 1st vertex has index '1', not '0'.
        """
        vertices, faces = mesh
        with open(output_path, 'w') as f:
            for vertex in vertices:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in faces:
                f.write('f %d %d %d\n' % tuple(face))


class JsonSaver:
    def __init__(self) -> None:
        self.extension = '.json'

    def __call__(self, flame_params: Dict[str, List[float]], output_path: str) -> None:
        with open(output_path, "w") as out:
            json.dump(flame_params, out)


def get_output_path(input_image_path: str, outputs_folder: str, type_of_output: str, extension: str) -> str:
    """
    Returns the output_path in outputs_folder that matches input_image_filename(tail), extended with the 'type of
    output' substring, with corresponding extension.
    """
    input_filename = os.path.split(input_image_path)[1]
    output_path_wo_ext = os.path.join(
        outputs_folder,
        f'{os.path.splitext(input_filename)[0]}_{type_of_output}'
    )
    return output_path_wo_ext + extension
# endregion



def _load_flame_faces_zero_based() -> np.ndarray:
    faces_path = get_relative_path(
        os.path.join(FLAME_IDICES_DIR, "../flame_mesh_faces.pt"), __file__
    )
    faces = torch.load(faces_path)
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    return faces.astype(np.int32)


def _get_face_vertex_set() -> np.ndarray:
    candidates = [
        "face_vertices.npy",
        "face_verts.npy",
        "face_indices.npy",
    ]
    for name in candidates:
        p = get_relative_path(os.path.join(FLAME_IDICES_DIR, name), __file__)
        if os.path.exists(p):
            v = np.load(p)
            return np.unique(v.astype(np.int32))

    try:
        edges_path = get_relative_path(os.path.join(FLAME_IDICES_DIR, "face_edges.npy"), __file__)
        edges = np.load(edges_path).astype(np.int32)  # [E,2]
        return np.unique(edges.reshape(-1))
    except Exception:
        return None

def get_depth(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    H, W = image.shape[:2]

    proj = predictions["projected_vertices"].squeeze().detach().cpu().numpy().astype(np.float32)  # [N,2]
    verts3d = predictions["3d_vertices"].detach().cpu().numpy().astype(np.float32)                # [N,3]
    faces = _load_flame_faces_zero_based()                                                         # [F,3]

    z = verts3d[:, 2]
    depth_per_vert = -z if np.median(z) < 0 else z

    depth = np.full((H, W), np.inf, dtype=np.float32)
    mask  = np.zeros((H, W), dtype=np.uint8)

    for f in faces:
        tri = proj[f, :]  # [[x,y],[x,y],[x,y]]
        if not np.all(np.isfinite(tri)):
            continue

        rr, cc = polygon(tri[:, 1], tri[:, 0], shape=(H, W))
        if rr.size == 0:
            continue

        ax, ay = tri[0]; bx, by = tri[1]; cx, cy = tri[2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-8:
            continue

        px = cc + 0.5
        py = rr + 0.5
        alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        beta  = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        gamma = 1.0 - alpha - beta

        z_tri = depth_per_vert[f]  # [3]
        z_interp = alpha * z_tri[0] + beta * z_tri[1] + gamma * z_tri[2]

        cur = depth[rr, cc]
        closer = z_interp < cur
        if np.any(closer):
            depth[rr[closer], cc[closer]] = z_interp[closer]
            mask[rr[closer], cc[closer]] = 255

    valid = (mask > 0) & np.isfinite(depth)
    if np.any(valid):
        dmin, dmax = depth[valid].min(), depth[valid].max()
        denom = max(dmax - dmin, 1e-8)
        dep01 = np.zeros_like(depth, dtype=np.float32)
        dep01[valid] = (depth[valid] - dmin) / denom
    else:
        dep01 = np.zeros_like(depth, dtype=np.float32)

    depth_u8 = (dep01 * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)  # BGR
    return depth_color


def get_face_region(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    H, W = image.shape[:2]
    proj  = predictions["projected_vertices"].squeeze().detach().cpu().numpy().astype(np.float32)  # [N,2]
    faces = _load_flame_faces_zero_based()                                                          # [F,3]

    face_vs = _get_face_vertex_set()
    if face_vs is not None and face_vs.size > 0:
        keep = np.isin(faces, face_vs).all(axis=1)
        faces_sel = faces[keep]
    else:
        faces_sel = faces

    mask = np.zeros((H, W), dtype=np.uint8)
    for f in faces_sel:
        tri = proj[f, :].astype(np.float32)
        rr, cc = polygon(tri[:, 1], tri[:, 0], shape=(H, W))
        if rr.size > 0:
            mask[rr, cc] = 255

    mask_bgr = cv2.merge([mask, mask, mask])
    return mask_bgr


def _get_face_vertex_set():
    candidates = ["face_vertices.npy", "face_verts.npy", "face_indices.npy"]
    for name in candidates:
        p = get_relative_path(os.path.join(FLAME_IDICES_DIR, name), __file__)
        if os.path.exists(p):
            v = np.load(p)
            return np.unique(v.astype(np.int32))

    try:
        edges_path = get_relative_path(os.path.join(FLAME_IDICES_DIR, "face_edges.npy"), __file__)
        edges = np.load(edges_path).astype(np.int32)  # [E,2]
        return np.unique(edges.reshape(-1))
    except Exception:
        return None

from skimage.draw import polygon

def get_face_depth(predictions: Dict[str, torch.Tensor], image: np.ndarray) -> np.ndarray:
    H, W = image.shape[:2]

    proj = predictions["projected_vertices"].squeeze().detach().cpu().numpy().astype(np.float32)  # [N,2]
    verts3d = predictions["3d_vertices"].detach().cpu().numpy().astype(np.float32)                # [N,3]
    faces_all = _load_flame_faces_zero_based()                                                     # [F,3]

    face_vs = _get_face_vertex_set()
    if face_vs is not None and face_vs.size > 0:
        keep = np.isin(faces_all, face_vs).all(axis=1)
        faces = faces_all[keep]
    else:
        faces = faces_all

    z = verts3d[:, 2]
    depth_per_vert = -z if np.median(z) < 0 else z

    # z-buffer
    depth = np.full((H, W), -np.inf, dtype=np.float32)
    mask  = np.zeros((H, W), dtype=np.uint8)

    for f in faces:
        tri = proj[f, :]  # [[x,y],[x,y],[x,y]]
        if not np.all(np.isfinite(tri)):
            continue

        rr, cc = polygon(tri[:, 1], tri[:, 0], shape=(H, W))
        if rr.size == 0:
            continue

        ax, ay = tri[0]; bx, by = tri[1]; cx, cy = tri[2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-8:
            continue

        px = cc + 0.5
        py = rr + 0.5
        alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        beta  = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        gamma = 1.0 - alpha - beta

        z_tri = depth_per_vert[f]  # [3]
        z_interp = alpha * z_tri[0] + beta * z_tri[1] + gamma * z_tri[2]

        cur = depth[rr, cc]
        closer = z_interp > cur
        if np.any(closer):
            depth[rr[closer], cc[closer]] = z_interp[closer]
            mask[rr[closer], cc[closer]] = 255 

    dep01 = np.zeros_like(depth, dtype=np.float32)
    valid = (mask > 0) & np.isfinite(depth)
    if np.any(valid):
        dmin, dmax = depth[valid].min(), depth[valid].max()
        dep01[valid] = (depth[valid] - dmin) / max(dmax - dmin, 1e-8)

    depth_u8   = (dep01 * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    depth_color[mask == 0] = 0
    return depth_color