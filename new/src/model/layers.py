"""网络层: EdgeConv / FeatureExtraction / Decoder。

FeatureExtraction 是 DGCNN 风格的特征提取器:
  conv1(input_dim) -> conv2 -> conv3(concat), 每层在 k 近邻上做 EdgeConv。
Decoder 是把每个点的特征向量映射成 3 维位移向量(velocity)。
"""
from typing import Optional

import torch
from torch import nn


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """批量成对平方欧氏距离, shape (B, N, M)。

    用 ||x||² + ||y||² - 2xy 展开, clamp 防止浮点误差导致负值。
    """
    x_norm = (x * x).sum(dim=-1, keepdim=True)
    y_norm = (y * y).sum(dim=-1).unsqueeze(1)
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.transpose(1, 2))
    return dist.clamp_min_(0.0)


def knn_indices(x: torch.Tensor, y: torch.Tensor, k: int, offset: int = 0) -> torch.Tensor:
    """返回 y 中每个 x 点的 k 近邻下标, shape (B, N, k)。offset 跳过前 offset 个(如跳过自身)。"""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("x and y must have shape (B, N, C)/(B, M, C)")
    max_neighbors = y.shape[1]
    requested = min(k + offset, max_neighbors)
    dist = pairwise_squared_distance(x, y)
    idx = torch.topk(dist, k=requested, dim=-1, largest=False, sorted=True).indices
    return idx[:, :, offset:]


class EdgeConv(nn.Module):
    """EdgeConv: 在 k 近邻图上做 edge feature 聚合。

    message = MLP([x_i, x_j - x_i]) 后对每个 dst 做 mean 聚合, 再加残差线性项。
    """

    def __init__(self, in_channels: int, out_channels: int, activation: Optional[str] = "ReLU"):
        super().__init__()
        if activation == "ReLU":
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels), nn.ReLU(),
                nn.Linear(out_channels, out_channels), nn.ReLU(),
            )
            self.lin = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels), nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError(f"unsupported activation: {activation}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        x_i, x_j = x[dst], x[src]
        message = self.mlp(torch.cat([x_i, x_j - x_i], dim=1))

        # 对同一 dst 的多条消息做 mean 聚合
        output = torch.zeros((x.shape[0], message.shape[1]), dtype=message.dtype, device=message.device)
        count = torch.zeros_like(output)
        output.index_add_(0, dst, message)
        count.index_add_(0, dst, torch.ones_like(message))
        output = output / count.clamp_min(1.0)
        return output + self.lin(x)


class DynamicEdgeConv(EdgeConv):
    """DynamicEdgeConv: 每层重新计算 k 近邻(动态图)。这里只是命名区分, 实现复用 EdgeConv。"""
    pass


class FeatureExtraction(nn.Module):
    """DGCNN 风格特征提取: 三层 EdgeConv, 最后一层输入是前两层特征拼接。"""

    def __init__(self, k: int = 32, input_dim: int = 3, embedding_dim: int = 512):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.conv1 = DynamicEdgeConv(input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            activation=None,
        )

    def _build_knn_edges(self, x: torch.Tensor) -> torch.Tensor:
        """对 (B, N, C) 点云建 k 近邻图, 返回 edge_index shape (2, E)。"""
        b, n, _ = x.shape
        if n < 2:
            raise ValueError("EdgeConv requires at least two points")
        k = min(self.k, n - 1)
        knn = knn_indices(x, x, k=k, offset=1)  # 跳过自身

        base = torch.arange(b, device=x.device, dtype=torch.long).view(b, 1, 1) * n
        src = (knn + base).reshape(-1)
        dst = (
            torch.arange(n, device=x.device, dtype=torch.long).view(1, n, 1).expand(b, n, k) + base
        ).reshape(-1)
        return torch.stack([src, dst], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        edges = self._build_knn_edges(x)
        x1 = self.conv1(x.reshape(b * n, -1), edges).reshape(b, n, -1)

        edges = self._build_knn_edges(x1)
        x2 = self.conv2(x1.reshape(b * n, -1), edges).reshape(b, n, -1)

        edges = self._build_knn_edges(x2)
        x3 = self.conv3(torch.cat([x1, x2], dim=-1).reshape(b * n, -1), edges)
        return x3.reshape(b, n, -1)


class Decoder(nn.Module):
    """把每个点的特征 -> 3 维位移向量(velocity field 的方向)。"""

    def __init__(self, z_dim: int, out_dim: int, hidden_size: int):
        super().__init__()
        self.out_dim = out_dim
        self.lin_1 = nn.Linear(z_dim, z_dim)
        self.bn_1 = nn.BatchNorm1d(z_dim)
        self.lin_2 = nn.Linear(z_dim, hidden_size)
        self.bn_2 = nn.BatchNorm1d(hidden_size)
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        net = self.dropout(self.act(self.bn_1(self.lin_1(c))))
        net = self.dropout(self.act(self.bn_2(self.lin_2(net))))
        if self.out_dim == 1:
            # 标量输出路径: max-pool 成一个全局分数后 sigmoid
            raise NotImplementedError("scalar output path not used by VelocityModule")
        return self.lin_3(net)
