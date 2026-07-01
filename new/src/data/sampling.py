"""点云采样与几何运算的纯 numpy/torch 实现。

这里把原来散在 utils.py 里的采样、旋转、FPS、KNN 集中到一个文件,
方便单独测试和日后替换成 pytorch3d 等高性能实现。
"""
from typing import Dict, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation


# ===================== 数组校验 =====================

def assert_ndarray(
    arr,
    name: str = "arr",
    shape: Optional[Tuple[int, ...]] = None,
    dtype=None,
) -> None:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray, got {type(arr)}")
    if shape is not None:
        if len(shape) != arr.ndim:
            raise ValueError(f"{name}: expected ndim {len(shape)}, got {arr.ndim}")
        for axis, (expected, actual) in enumerate(zip(shape, arr.shape)):
            if expected is not None and expected > 0 and expected != actual:
                raise ValueError(
                    f"{name} shape mismatch at axis {axis}: expected {expected}, got {actual}"
                )
    if dtype is not None and not np.issubdtype(arr.dtype, dtype):
        raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")


# ===================== 网格表面采样 =====================

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    mask: Optional[ndarray] = None,
    face_index: Optional[ndarray] = None,
    random_lengths: Optional[ndarray] = None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """按面积加权在三角网格表面均匀采样点。

    返回 (samples, original_face_index, selected_face_index, random_lengths)。
    后三者用于 sample_barycentric 复用同一采样参数(如对 noisy/clean 用同一组面)。
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")

    original_face_indices = np.arange(len(faces))
    if mask is not None:
        original_face_indices = original_face_indices[mask]
        faces = faces[mask]
    if len(faces) == 0:
        raise ValueError("cannot sample from an empty face set")

    # 按三角形面积做累积分布, 用逆变换采样选面
    if face_index is None:
        offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        face_weight = np.linalg.norm(np.cross(offset_0, offset_1, axis=-1), axis=-1)
        weight_cum = np.cumsum(face_weight, axis=0)
        if weight_cum[-1] <= 0:
            raise ValueError("mesh has zero total face area")
        face_pick = np.random.rand(num_samples) * weight_cum[-1]
        selected_face_index = np.searchsorted(weight_cum, face_pick)
    else:
        selected_face_index = face_index

    original_face_index = original_face_indices[selected_face_index]

    # 在选中三角形内用重心坐标采样
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]].copy()
    tri_vectors -= tri_origins[:, None, :]
    tri_origins = tri_origins[selected_face_index]
    tri_vectors = tri_vectors[selected_face_index]

    if random_lengths is None:
        random_lengths = np.random.rand(len(tri_vectors), 2, 1)

    # 若 u+v>1 则翻转, 保证落在三角形内
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    return sample_vector + tri_origins, original_face_index, selected_face_index, random_lengths


def sample_barycentric(
    vertex_group: ndarray,
    faces: ndarray,
    face_index: ndarray,
    random_lengths: ndarray,
) -> ndarray:
    """用同一组重心坐标对另一组顶点(如顶点法向/颜色)插值采样。"""
    v_origins = vertex_group[faces[face_index, 0]]
    v_vectors = vertex_group[faces[face_index, 1:]].copy()
    v_vectors -= v_origins[:, None, :]
    return (v_vectors * random_lengths).sum(axis=1) + v_origins


def sample_vertex_groups(
    vertices: ndarray,
    faces: ndarray,
    num_samples: int,
    num_vertex_samples: Optional[int] = None,
    face_mask: Optional[ndarray] = None,
) -> Tuple[ndarray, Dict[str, ndarray]]:
    """采样 num_samples 个点 = (num_vertex_samples 个网格顶点) + (表面采样补足)。

    返回 (sampled_points, params), params 可传给 sample_surface 复用。
    """
    if num_vertex_samples is None:
        num_vertex_samples = 0
    if num_vertex_samples > num_samples:
        raise ValueError(f"num_vertex_samples {num_vertex_samples} > num_samples {num_samples}")

    # 选顶点子集
    if face_mask is not None:
        vertex_mask = np.unique(faces[face_mask])
    else:
        vertex_mask = np.arange(vertices.shape[0])
    count = min(num_vertex_samples, vertex_mask.shape[0])
    perm = vertex_mask[np.random.permutation(vertex_mask.shape[0])[:count]]

    face_vertices, original_face_index, face_index, random_lengths = sample_surface(
        num_samples=num_samples - len(perm),
        vertices=vertices,
        faces=faces,
        mask=face_mask,
    )
    sampled_vertices = np.concatenate([vertices[perm], face_vertices], axis=0)
    return sampled_vertices, {
        "perm": perm,
        "original_face_index": original_face_index,
        "face_index": face_index,
        "random_lengths": random_lengths,
    }


# ===================== 随机旋转 =====================

def random_euler_rotation(
    batch_size: int = 1,
    x_range=(0.0, 0.0),
    y_range=(0.0, 0.0),
    z_range=(0.0, 0.0),
    degrees: bool = True,
) -> ndarray:
    """生成 batch_size 个 4x4 仿射矩阵, 含独立随机欧拉旋转。"""
    x_deg = np.random.uniform(*x_range, size=batch_size)
    y_deg = np.random.uniform(*y_range, size=batch_size)
    z_deg = np.random.uniform(*z_range, size=batch_size)
    rot = Rotation.from_euler("ZYX", np.vstack([z_deg, y_deg, x_deg]).T, degrees=degrees)
    mats = rot.as_matrix().astype(np.float32)

    mats4 = np.zeros((batch_size, 4, 4), dtype=np.float32)
    mats4[:, :3, :3] = mats
    mats4[:, 3, 3] = 1.0
    return mats4


# ===================== 点云归一化 =====================

def normalize_to_unit_sphere(pc: ndarray) -> Tuple[ndarray, ndarray, float]:
    """归一化到单位球(中心移到原点, 最远点距离=1)。返回 (normalized, center, scale)。"""
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    pc_centered = pc - center
    scale = float(np.sqrt((pc_centered ** 2).sum(axis=1)).max())
    if scale < 1e-12:
        return pc_centered, center, scale
    return pc_centered / scale, center, scale
