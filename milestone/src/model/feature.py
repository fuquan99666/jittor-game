from typing import Optional

import torch
from torch import nn


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return batched squared Euclidean distances, shape (B, N, M)."""
    x_norm = (x * x).sum(dim=-1, keepdim=True)
    y_norm = (y * y).sum(dim=-1).unsqueeze(1)
    distance = x_norm + y_norm - 2.0 * torch.matmul(x, y.transpose(1, 2))
    return distance.clamp_min_(0.0)


def get_knn_idx(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    offset: int = 0,
) -> torch.Tensor:
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("x and y must have shape (B, N, C)/(B, M, C)")
    max_neighbors = y.shape[1]
    requested = k + offset
    if requested > max_neighbors:
        requested = max_neighbors
    distances = pairwise_squared_distance(x, y)
    indices = torch.topk(
        distances,
        k=requested,
        dim=-1,
        largest=False,
        sorted=True,
    ).indices
    return indices[:, :, offset:]


class EdgeConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = "ReLU",
    ):
        super().__init__()
        if activation == "ReLU":
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
            )
            self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
            )
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError(f"unsupported activation: {activation}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        x_i = x[dst]
        x_j = x[src]
        message = self.mlp(torch.cat([x_i, x_j - x_i], dim=1))

        output = torch.zeros(
            (x.shape[0], message.shape[1]),
            dtype=message.dtype,
            device=message.device,
        )
        count = torch.zeros_like(output)
        output.index_add_(0, dst, message)
        count.index_add_(0, dst, torch.ones_like(message))
        output = output / count.clamp_min(1.0)
        return output + self.lin(x)


class DynamicEdgeConv(EdgeConv):
    pass


class FeatureExtraction(nn.Module):
    def __init__(
        self,
        k: int = 32,
        input_dim: int = 0,
        embedding_dim: int = 512,
        distance_estimation: bool = False,
    ):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.distance_estimation = distance_estimation

        self.conv1 = DynamicEdgeConv(input_dim, embedding_dim // 8)
        self.conv2 = DynamicEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = DynamicEdgeConv(
            embedding_dim // 8 + embedding_dim // 4,
            embedding_dim,
            activation=None,
        )

    def get_edge_index(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, _ = x.shape
        if point_count < 2:
            raise ValueError("EdgeConv requires at least two points")
        k = min(self.k, point_count - 1)
        knn_idx = get_knn_idx(x, x, k=k, offset=1)

        base = (
            torch.arange(batch_size, device=x.device, dtype=torch.long)
            .view(batch_size, 1, 1)
            * point_count
        )
        src = (knn_idx + base).reshape(-1)
        dst = (
            torch.arange(point_count, device=x.device, dtype=torch.long)
            .view(1, point_count, 1)
            .expand(batch_size, point_count, k)
            + base
        ).reshape(-1)
        return torch.stack([src, dst], dim=0)

    @staticmethod
    def normalize_patch(pcl: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt((pcl**2).sum(dim=-1, keepdim=True))
        scale = scale.max(dim=-2, keepdim=True).values
        return pcl / (scale + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, _ = x.shape
        if self.distance_estimation:
            x = self.normalize_patch(x)

        edge_index = self.get_edge_index(x)
        x1 = self.conv1(x.reshape(batch_size * point_count, -1), edge_index)
        x1 = x1.reshape(batch_size, point_count, -1)

        edge_index = self.get_edge_index(x1)
        x2 = self.conv2(x1.reshape(batch_size * point_count, -1), edge_index)
        x2 = x2.reshape(batch_size, point_count, -1)

        edge_index = self.get_edge_index(x2)
        combined = torch.cat([x1, x2], dim=-1)
        x3 = self.conv3(
            combined.reshape(batch_size * point_count, -1),
            edge_index,
        )
        return x3.reshape(batch_size, point_count, -1)


class Decoder(nn.Module):
    def __init__(self, z_dim: int, dim: int, out_dim: int, hidden_size: int):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size

        self.lin_1 = nn.Linear(z_dim, z_dim)
        self.bn_1_out = nn.BatchNorm1d(z_dim)
        self.lin_2 = nn.Linear(z_dim, hidden_size)
        self.bn_2_out = nn.BatchNorm1d(hidden_size)
        self.lin_3 = nn.Linear(hidden_size, out_dim)
        self.actvn_out = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        c: torch.Tensor,
        batch_size: Optional[int] = None,
        point_count: Optional[int] = None,
    ) -> torch.Tensor:
        net = self.dropout(self.actvn_out(self.bn_1_out(self.lin_1(c))))
        net = self.dropout(self.actvn_out(self.bn_2_out(self.lin_2(net))))

        if self.out_dim == 1:
            if batch_size is None or point_count is None:
                raise ValueError("batch_size and point_count are required for scalar output")
            net = net.reshape(batch_size, point_count, -1)
            net = net.max(dim=1, keepdim=True).values
            return torch.sigmoid(self.lin_3(net))
        return self.lin_3(net)
