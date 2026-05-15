import torch
from torch import nn
import spconv.pytorch as spconv


MANO_POSE_DIM = 48
MANO_SHAPE_DIM = 10
MANO_PARAM_DIM = MANO_POSE_DIM + MANO_SHAPE_DIM
EDGE_THRESHOLD = 0.1
STAGE_CHANNELS = (32, 64, 128, 256)
HEAD_HIDDEN_DIM = 512


def dense_to_sparse_tensor(edge_map, threshold):
    batch_size, _, height, width = edge_map.shape
    mask = edge_map.squeeze(1) > threshold
    batch_idx, y_idx, x_idx = mask.nonzero(as_tuple=True)
    coords = torch.stack([batch_idx, y_idx, x_idx], dim=1).int()
    features = edge_map.squeeze(1)[batch_idx, y_idx, x_idx].unsqueeze(1)
    return spconv.SparseConvTensor(features, coords, (height, width), batch_size)


def build_downsample_block(in_channels, out_channels, indice_key):
    return spconv.SparseSequential(
        spconv.SparseConv2d(in_channels, out_channels, kernel_size=3, stride=2,
                            padding=1, bias=False, indice_key=f"{indice_key}_down"),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
        spconv.SubMConv2d(out_channels, out_channels, kernel_size=3, padding=1,
                          bias=False, indice_key=f"{indice_key}_sub"),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
    )


class SparseEdgeBackbone(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.stem = spconv.SparseSequential(
            spconv.SparseConv2d(1, channels[0], kernel_size=3, stride=2, padding=1,
                                bias=False, indice_key="stem"),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        self.stage1 = build_downsample_block(channels[0], channels[1], "stage1")
        self.stage2 = build_downsample_block(channels[1], channels[2], "stage2")
        self.stage3 = build_downsample_block(channels[2], channels[3], "stage3")
        self.output_channels = channels[3]

    def forward(self, sparse_input):
        x = self.stem(sparse_input)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


def global_mean_pool_sparse(sparse_tensor, batch_size):
    coords = sparse_tensor.indices
    features = sparse_tensor.features
    feature_dim = features.size(1)

    pooled_sum = torch.zeros(batch_size, feature_dim, device=features.device, dtype=features.dtype)
    counts = torch.zeros(batch_size, 1, device=features.device, dtype=features.dtype)

    batch_indices = coords[:, 0].long()
    pooled_sum.index_add_(0, batch_indices, features)
    counts.index_add_(0, batch_indices, torch.ones_like(features[:, :1]))

    return pooled_sum / counts.clamp(min=1.0)


class MANORegressorHead(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, MANO_PARAM_DIM),
        )

    def forward(self, pooled_features):
        params = self.mlp(pooled_features)
        return params[:, :MANO_POSE_DIM], params[:, MANO_POSE_DIM:]


class MobileMaskHand(nn.Module):
    def __init__(self, edge_threshold=EDGE_THRESHOLD):
        super().__init__()
        self.edge_threshold = edge_threshold
        self.backbone = SparseEdgeBackbone(STAGE_CHANNELS)
        self.head = MANORegressorHead(self.backbone.output_channels, HEAD_HIDDEN_DIM)

    def forward(self, edge_map):
        batch_size = edge_map.size(0)
        sparse_input = dense_to_sparse_tensor(edge_map, self.edge_threshold)
        sparse_features = self.backbone(sparse_input)
        pooled = global_mean_pool_sparse(sparse_features, batch_size)
        pose, shape = self.head(pooled)
        return {"theta": pose, "beta": shape}
