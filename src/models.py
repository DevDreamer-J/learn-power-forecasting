"""模型定义：PatchTST 与 DLinear，均带 RevIN 实例归一化。

设计说明：
- 输入 [B, L, C]，第 0 通道为功率，其余通道为时间特征（日/周/年周期编码）。
- PatchTST 通道独立：每个通道单独 patch 后过同一个 Transformer 编码器，
  输出每个通道的未来 horizon 点，但损失只取第 0 通道（功率）。
- RevIN 按窗口逐通道标准化再还原，缓解负载序列的分布漂移。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def revin_norm(x: torch.Tensor, eps: float = 1e-5):
    """RevIN 归一化。x: [B, L, C] -> (归一化后的 x, 均值 [B,1,C], 标准差 [B,1,C])"""
    mean = x.mean(dim=1, keepdim=True)
    std = (x.var(dim=1, keepdim=True, unbiased=False) + eps).sqrt()
    return (x - mean) / std, mean, std


class PatchTST(nn.Module):
    """PatchTST：把序列切成重叠 patch，经 Transformer 编码后线性映射到预测 horizon。"""

    def __init__(self, lookback: int, horizon: int, n_channels: int, target_idx: int = 0,
                 patch_len: int = 16, stride: int = 8, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 3, d_ff: int = 256, dropout: float = 0.2):
        super().__init__()
        self.target_idx = target_idx
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model * self.n_patches, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, L, C]
        B, L, C = x.shape
        x, mean, std = revin_norm(x)                          # [B, L, C]

        # 通道独立：把 C 个通道摊到 batch 维
        z = x.permute(0, 2, 1).reshape(B * C, L)              # [B*C, L]
        patches = z.unfold(1, self.patch_len, self.stride)    # [B*C, P, patch_len]
        h = self.dropout(self.embed(patches) + self.pos_embed)
        h = self.encoder(h)                                   # [B*C, P, D]

        out = self.head(h.flatten(1))                         # [B*C, H]
        out = out.reshape(B, C, -1)                           # [B, C, H]
        pred = out[:, self.target_idx]                        # [B, H]

        # 用目标通道自身的 RevIN 统计量还原量纲
        pred = pred * std[:, :, self.target_idx] + mean[:, :, self.target_idx]
        return pred


class MovingAvg(nn.Module):
    """滑动平均，用于把序列分解为趋势项 + 季节项（DLinear）。"""

    def __init__(self, kernel: int = 25):
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, L, C]
        front = x[:, :1, :].repeat(1, (self.kernel - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel - 1 - (self.kernel - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return F.avg_pool1d(x.permute(0, 2, 1), kernel_size=self.kernel,
                            stride=1).permute(0, 2, 1)


class DLinear(nn.Module):
    """DLinear：趋势/季节分解 + 每通道一个线性层，是必须打赢的强 baseline。"""

    def __init__(self, lookback: int, horizon: int, n_channels: int,
                 target_idx: int = 0, kernel: int = 25):
        super().__init__()
        self.target_idx = target_idx
        self.avg = MovingAvg(kernel)
        self.linear_seasonal = nn.ModuleList(
            nn.Linear(lookback, horizon) for _ in range(n_channels))
        self.linear_trend = nn.ModuleList(
            nn.Linear(lookback, horizon) for _ in range(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, L, C]
        x, mean, std = revin_norm(x)
        trend = self.avg(x)
        seasonal = x - trend
        B, _, C = x.shape
        out = torch.stack(
            [self.linear_seasonal[c](seasonal[:, :, c])
             + self.linear_trend[c](trend[:, :, c])
             for c in range(C)], dim=-1)                      # [B, H, C]
        pred = out[:, :, self.target_idx]                     # [B, H]
        pred = pred * std[:, :, self.target_idx] + mean[:, :, self.target_idx]
        return pred
