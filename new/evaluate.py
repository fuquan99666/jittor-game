#!/usr/bin/env python
"""点云去噪评测脚本: 计算 Chamfer Distance (CD) 和 Point-to-Surface (P2S)。

评分:
  单指标 score = clamp(100 * (1 - val_pred / val_noisy), 0, 100)
  final = 0.5*CD + 0.5*P2S  (有 mesh 时), 否则只用 CD
  缺失预测记 0 分

期望文件名: pred=denoised.npy, gt=clean.npy, noisy=noisy.npy,
           mesh=models/model_normalized.obj (相对 model_id)

用法:
  python evaluate.py --pred_dir results --gt_dir <gt> --noisy_dir <noisy> [--mesh_dir <mesh>]
"""
import argparse
import glob
import os
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="point_cloud_utils")

import numpy as np

try:
    import point_cloud_utils as pcu
    HAS_PCU = True
except ImportError:
    HAS_PCU = False

from scipy.spatial import cKDTree


# ===================== IO =====================

def load_pointcloud(path: str) -> np.ndarray:
    """加载点云, 支持 .npy 和 .xyz 文本。"""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts, dtype=np.float64)


def load_mesh_vf(path: str):
    """加载网格, 返回 (vertices, faces)。优先 pcu, 退而用 trimesh。"""
    if not os.path.exists(path):
        return None, None
    if HAS_PCU:
        v, f = pcu.load_mesh_vf(path)
        return v.astype(np.float64), f.astype(np.int32)
    try:
        import trimesh
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int32)
    except ImportError:
        return None, None


# ===================== 指标 =====================

def normalize_to_unit_sphere(pc: np.ndarray):
    """归一化到单位球。返回 (normalized, center, scale)。"""
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    pc_centered = pc - center
    scale = np.sqrt((pc_centered ** 2).sum(axis=1)).max()
    if scale < 1e-12:
        return pc_centered, center, scale
    return pc_centered / scale, center, scale


def chamfer_distance(pc_a: np.ndarray, pc_b: np.ndarray, normalize: bool = True) -> float:
    """CD = mean_a min_b ||a-b||² + mean_b min_a ||b-a||²。

    normalize=True 时以 pc_b(参考)归一化到单位球, pc_a 施加相同变换。
    用 scipy cKDTree 加速最近邻查询。
    """
    if normalize:
        pc_b, center, scale = normalize_to_unit_sphere(pc_b)
        if scale < 1e-12:
            return 0.0
        pc_a = (pc_a - center) / scale

    dist_a2b, _ = cKDTree(pc_b).query(pc_a, k=1)
    dist_b2a, _ = cKDTree(pc_a).query(pc_b, k=1)
    return float((dist_a2b ** 2).mean() + (dist_b2a ** 2).mean())


def point_to_surface_distance(pc, mesh_v, mesh_f, normalize_ref_pc=None):
    """P2S: 每个点到网格表面最近距离²的均值。

    有 pcu 时用 BVH 加速的精确最近点; 否则退化为网格顶点近似(cKDTree)。
    """
    if mesh_v is None or mesh_f is None:
        return None

    vertices = mesh_v.copy()
    if normalize_ref_pc is not None:
        center = (normalize_ref_pc.max(axis=0) + normalize_ref_pc.min(axis=0)) / 2.0
        centered = normalize_ref_pc - center
        scale = np.sqrt((centered ** 2).sum(axis=1)).max()
        if scale < 1e-12:
            return 0.0
        pc = (pc - center) / scale
        vertices = (vertices - center) / scale

    if HAS_PCU:
        dists, _, _ = pcu.closest_points_on_mesh(
            pc.astype(np.float32), vertices.astype(np.float32), mesh_f
        )
        return float((dists ** 2).mean())

    tree = cKDTree(vertices)
    dists, _ = tree.query(pc, k=1)
    return float((dists ** 2).mean())


def metric_to_score(val_pred: float, val_noisy: float) -> float:
    """score = clamp(100 * (1 - pred/noisy), 0, 100)。"""
    if val_noisy < 1e-15:
        return 100.0 if val_pred < 1e-15 else 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - val_pred / val_noisy)))


# ===================== 文件扫描 =====================

def find_samples(base_dir: str, filename: str) -> dict:
    """递归扫描目录, 返回 {relative_key: filepath}。"""
    samples = {}
    for path in sorted(glob.glob(os.path.join(base_dir, "**", filename), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), base_dir)
        samples[rel] = path
    return samples


def find_meshes(mesh_dir: str, data_name: str = "models/model_normalized.obj") -> dict:
    """扫描网格目录, key 为 model_id(去掉 data_name 后缀的相对路径)。"""
    meshes = {}
    for path in sorted(glob.glob(os.path.join(mesh_dir, "**", data_name), recursive=True)):
        p = path
        for _ in range(len(data_name.split("/"))):
            p = os.path.dirname(p)
        meshes[os.path.relpath(p, mesh_dir)] = path
    return meshes


# ===================== 单样本评测 =====================

def evaluate_single(args_tuple):
    key, pred_path, gt_path, noisy_path, mesh_path = args_tuple
    pc_pred = load_pointcloud(pred_path)
    pc_gt = load_pointcloud(gt_path)
    pc_noisy = load_pointcloud(noisy_path)

    cd_pred = chamfer_distance(pc_pred, pc_gt, normalize=True)
    cd_noisy = chamfer_distance(pc_noisy, pc_gt, normalize=True)
    cd_score = metric_to_score(cd_pred, cd_noisy)

    p2s_pred = p2s_noisy = p2s_score = None
    if mesh_path is not None:
        mv, mf = load_mesh_vf(mesh_path)
        if mv is not None:
            p2s_pred = point_to_surface_distance(pc_pred, mv, mf, normalize_ref_pc=pc_gt)
            p2s_noisy = point_to_surface_distance(pc_noisy, mv, mf, normalize_ref_pc=pc_gt)
            if p2s_pred is not None and p2s_noisy is not None:
                p2s_score = metric_to_score(p2s_pred, p2s_noisy)

    return (key, cd_pred, cd_noisy, cd_score, p2s_pred, p2s_noisy, p2s_score)


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="点云降噪评测脚本")
    parser.add_argument("--pred_dir", type=str, required=True, help="降噪结果目录")
    parser.add_argument("--gt_dir", type=str, required=True, help="干净点云目录")
    parser.add_argument("--noisy_dir", type=str, required=True, help="含噪点云目录")
    parser.add_argument("--mesh_dir", type=str, default="", help="原始网格目录(用于 P2S, 可选)")
    parser.add_argument("--mesh_data_name", type=str, default="models/model_normalized.obj")
    parser.add_argument("--pred_filename", type=str, default="denoised.npy")
    parser.add_argument("--gt_filename", type=str, default="clean.npy")
    parser.add_argument("--noisy_filename", type=str, default="noisy.npy")
    parser.add_argument("--workers", type=int, default=0, help="并行进程数 (0=自动)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    use_p2s = bool(args.mesh_dir)
    if use_p2s and not HAS_PCU:
        print("提示: point-cloud-utils 未安装, P2S 将用顶点近似。pip install point-cloud-utils 以获精确结果。")

    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 16)

    pred_samples = find_samples(args.pred_dir, args.pred_filename)
    gt_samples = find_samples(args.gt_dir, args.gt_filename)
    noisy_samples = find_samples(args.noisy_dir, args.noisy_filename)
    mesh_samples = find_meshes(args.mesh_dir, args.mesh_data_name) if use_p2s else {}

    print(f"后端: CD=scipy.cKDTree, P2S={'pcu(BVH)' if HAS_PCU else 'cKDTree(顶点近似)'}, workers={n_workers}")

    common_keys = sorted(set(pred_samples) & set(gt_samples) & set(noisy_samples))
    if not common_keys:
        print("错误: 未找到匹配的测试样本。")
        print(f"  pred={len(pred_samples)}, gt={len(gt_samples)}, noisy={len(noisy_samples)}")
        sys.exit(1)

    missing_pred = set(gt_samples) - set(pred_samples)
    if missing_pred:
        print(f"警告: {len(missing_pred)} 个样本缺少预测, 记 0 分。")

    tasks = [
        (key, pred_samples[key], gt_samples[key], noisy_samples[key],
         mesh_samples.get(key) if use_p2s else None)
        for key in common_keys
    ]

    print(f"开始评测 {len(tasks)} 个样本...")
    t0 = time.time()

    if n_workers > 1 and len(tasks) > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(evaluate_single, tasks)
    else:
        results = [evaluate_single(t) for t in tasks]

    cd_scores, p2s_scores = [], []
    cd_preds, cd_noisys = [], []
    p2s_preds, p2s_noisys = [], []

    for key, cd_pred, cd_noisy, cd_s, p2s_pred, p2s_noisy, p2s_s in results:
        cd_scores.append(cd_s)
        cd_preds.append(cd_pred)
        cd_noisys.append(cd_noisy)
        if p2s_s is not None:
            p2s_scores.append(p2s_s)
            p2s_preds.append(p2s_pred)
            p2s_noisys.append(p2s_noisy)
        if args.verbose:
            msg = f"  {key}  CD_score={cd_s:.2f}"
            if p2s_s is not None:
                msg += f"  P2S_score={p2s_s:.2f}"
            print(msg)

    for key in missing_pred:
        cd_scores.append(0.0)
        if use_p2s:
            p2s_scores.append(0.0)

    total = len(common_keys) + len(missing_pred)
    mean_cd = np.mean(cd_scores) if cd_scores else 0.0
    has_p2s = len(p2s_scores) > 0
    mean_p2s = np.mean(p2s_scores) if has_p2s else 0.0
    final = 0.5 * mean_cd + 0.5 * mean_p2s if has_p2s else mean_cd

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  点云降噪评测结果")
    print("=" * 65)
    print(f"  样本总数:   {total}")
    print(f"  有效预测:   {len(common_keys)}")
    print(f"  缺失预测:   {len(missing_pred)}")
    print(f"  并行进程:   {n_workers}")
    print(f"  耗时:       {elapsed:.1f}s")
    print("-" * 65)
    if cd_preds:
        print(f"  平均 CD_pred:   {np.mean(cd_preds):.8f}")
        print(f"  平均 CD_noisy:  {np.mean(cd_noisys):.8f}")
        print(f"  CD 得分:        {mean_cd:.2f} / 100.00")
    if has_p2s:
        print(f"  平均 P2S_pred:  {np.mean(p2s_preds):.8f}")
        print(f"  平均 P2S_noisy: {np.mean(p2s_noisys):.8f}")
        print(f"  P2S 得分:       {mean_p2s:.2f} / 100.00")
        print("-" * 65)
        print(f"  最终得分 (0.5*CD + 0.5*P2S):  {final:.2f} / 100.00")
    else:
        print("-" * 65)
        print(f"  最终得分 (CD):  {final:.2f} / 100.00")
        if not use_p2s:
            print("  (未提供 mesh_dir, P2S 未计算)")
    print("=" * 65)
    return final


if __name__ == "__main__":
    main()
