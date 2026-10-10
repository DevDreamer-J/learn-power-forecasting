"""全局配置：数据、窗口、模型与训练超参数。"""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # ---------- 数据 ----------
    data_path: Path = PROJECT_ROOT / "complete_data.csv"
    freq: str = "15min"          # 采样间隔
    points_per_day: int = 96     # 每天点数 24h * 4

    # ---------- 窗口 ----------
    lookback: int = 96 * 7       # 输入：过去 7 天（672 点）
    horizon: int = 96            # 输出：未来 1 天（96 点）

    # ---------- 数据划分（按时间切分）----------
    train_end: str = "2021-04-30 23:45:00"  # 训练段最后一帧（含）
    val_end: str = "2021-06-30 23:45:00"    # 验证段最后一帧（含），其后全部为测试段

    # ---------- 模型 ----------
    model: str = "patchtst"      # patchtst | dlinear
    # PatchTST 超参数
    patch_len: int = 16          # patch 长度
    patch_stride: int = 8        # patch 步长
    d_model: int = 128           # 编码维度
    n_heads: int = 8             # 注意力头数
    n_layers: int = 3            # 编码器层数
    d_ff: int = 256              # 前馈层维度
    dropout: float = 0.2

    # ---------- 训练 ----------
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5            # 早停耐心值（验证集连续多少轮不提升）
    amp: bool = True             # 混合精度训练
    num_workers: int = 0         # Windows 下 0 最稳
    seed: int = 42

    # ---------- 输出 ----------
    out_dir: Path = PROJECT_ROOT / "outputs"
