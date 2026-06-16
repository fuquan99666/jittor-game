from typing import Dict, Optional, Tuple

import numpy as np
from numpy import ndarray


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
            raise ValueError(
                f"{name}: expected shape length {len(shape)} but array ndim is {arr.ndim}"
            )
        for axis, (expected, actual) in enumerate(zip(shape, arr.shape)):
            if expected is not None and expected > 0 and expected != actual:
                raise ValueError(
                    f"{name} shape mismatch at axis {axis}: expected {expected}, got {actual}"
                )
    if dtype is not None and not np.issubdtype(arr.dtype, dtype):
        raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")


def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    mask: Optional[ndarray] = None,
    face_index: Optional[ndarray] = None,
    random_lengths: Optional[ndarray] = None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")

    original_face_indices = np.arange(len(faces))
    if mask is not None:
        original_face_indices = original_face_indices[mask]
        faces = faces[mask]
    if len(faces) == 0:
        raise ValueError("cannot sample from an empty face set")

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

    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]].copy()
    tri_vectors -= tri_origins[:, None, :]
    tri_origins = tri_origins[selected_face_index]
    tri_vectors = tri_vectors[selected_face_index]

    if random_lengths is None:
        random_lengths = np.random.rand(len(tri_vectors), 2, 1)

    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    return vertex_samples, original_face_index, selected_face_index, random_lengths


def sample_barycentric(
    vertex_group: ndarray,
    faces: ndarray,
    face_index: ndarray,
    random_lengths: ndarray,
) -> ndarray:
    v_origins = vertex_group[faces[face_index, 0]]
    v_vectors = vertex_group[faces[face_index, 1:]].copy()
    v_vectors -= v_origins[:, None, :]
    return (v_vectors * random_lengths).sum(axis=1) + v_origins


def sample_vertex_groups(
    vertices: ndarray,
    faces: ndarray,
    num_samples: int,
    num_vertex_samples: Optional[int] = None,
    vertex_normals: Optional[ndarray] = None,
    face_normals: Optional[ndarray] = None,
    vertex_groups: Optional[ndarray] = None,
    face_mask: Optional[ndarray] = None,
    deterministic_params: Optional[Dict[str, ndarray]] = None,
) -> Tuple[ndarray, Optional[ndarray], Optional[ndarray], Dict[str, ndarray]]:
    if num_vertex_samples is None:
        num_vertex_samples = 0
    if num_vertex_samples > num_samples:
        raise ValueError(
            f"num_vertex_samples cannot exceed num_samples: "
            f"{num_vertex_samples} > {num_samples}"
        )

    def get_mask_perm(mask_vertices: Optional[ndarray]) -> ndarray:
        if mask_vertices is None:
            vertex_mask = np.arange(vertices.shape[0])
        else:
            vertex_mask = np.unique(mask_vertices)
        count = min(num_vertex_samples, vertex_mask.shape[0])
        perm = np.random.permutation(vertex_mask.shape[0])
        return vertex_mask[perm[:count]]

    if vertex_groups is not None:
        if vertex_groups.ndim == 1:
            assert_ndarray(
                vertex_groups,
                name="vertex_groups",
                shape=(vertices.shape[0],),
            )
            vertex_groups = vertex_groups[:, None]
        else:
            assert_ndarray(
                vertex_groups,
                name="vertex_groups",
                shape=(vertices.shape[0], -1),
            )

    if deterministic_params is not None:
        perm = deterministic_params["perm"]
        original_face_index = deterministic_params["original_face_index"]
        face_index = deterministic_params["face_index"]
        random_lengths = deterministic_params["random_lengths"]
        face_vertices, original_face_index, face_index, random_lengths = sample_surface(
            num_samples=num_samples - len(perm),
            vertices=vertices,
            faces=faces,
            mask=face_mask,
            face_index=face_index,
            random_lengths=random_lengths,
        )
    else:
        if face_mask is not None:
            assert_ndarray(face_mask, name="face_mask", shape=(faces.shape[0],))
            perm = get_mask_perm(faces[face_mask])
        else:
            perm = get_mask_perm(None)
        face_vertices, original_face_index, face_index, random_lengths = sample_surface(
            num_samples=num_samples - len(perm),
            vertices=vertices,
            faces=faces,
            mask=face_mask,
        )

    sampled_vertices = np.concatenate([vertices[perm], face_vertices], axis=0)

    if vertex_normals is not None and face_normals is not None:
        sampled_normals = np.concatenate(
            [vertex_normals[perm], face_normals[original_face_index]], axis=0
        )
    else:
        sampled_normals = None

    if vertex_groups is not None:
        sampled_group_faces = sample_barycentric(
            vertex_group=vertex_groups,
            faces=faces,
            face_index=face_index,
            random_lengths=random_lengths,
        )
        sampled_vertex_groups = np.concatenate(
            [vertex_groups[perm], sampled_group_faces], axis=0
        )
    else:
        sampled_vertex_groups = None

    params = {
        "perm": perm,
        "original_face_index": original_face_index,
        "face_index": face_index,
        "random_lengths": random_lengths,
    }
    return sampled_vertices, sampled_normals, sampled_vertex_groups, params


def random_euler_rotation(
    batch_size: int = 1,
    x_range=(0, 0),
    y_range=(0, 0),
    z_range=(0, 0),
    degrees: bool = True,
    return_4x4: bool = True,
):
    from scipy.spatial.transform import Rotation

    x_deg = np.random.uniform(*x_range, size=batch_size)
    y_deg = np.random.uniform(*y_range, size=batch_size)
    z_deg = np.random.uniform(*z_range, size=batch_size)

    rot = Rotation.from_euler(
        "ZYX",
        np.vstack([z_deg, y_deg, x_deg]).T,
        degrees=degrees,
    )
    mats = rot.as_matrix().astype(np.float32)

    if not return_4x4:
        return mats

    mats4 = np.zeros((batch_size, 4, 4), dtype=np.float32)
    mats4[:, :3, :3] = mats
    mats4[:, 3, 3] = 1.0
    return mats4
