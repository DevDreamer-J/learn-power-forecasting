"""数据管道：读取 CSV、补齐时间网格、构造时间特征、标准化与滑动窗口数据集。"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class StandardScaler:
    """按训练段均值/方差做标准化，还原量纲时使用同一组统计量。"""

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def load_dataframe(cfg: Config) -> pd.DataFrame:
    """读取 CSV 并补齐 15 分钟完整时间网格，缺失点按时间线性插值。"""
    df = pd.read_csv(cfg.data_path, index_col=0, parse_dates=True)
    df.columns = ["load"]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    n_before = len(df)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=cfg.freq)
    df = df.reindex(full_index)
    n_missing = int(df["load"].isna().sum())
    if n_missing:
        df["load"] = df["load"].interpolate(method="time")
    print(f"[data] 原始 {n_before} 行 -> 补齐网格 {len(df)} 行，插值填补缺失 {n_missing} 点")
    print(f"[data] 时间范围: {df.index.min()} ~ {df.index.max()}")
    return df


def make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """日/周/年周期的正余弦编码，取值天然在 [-1, 1]，无需再标准化。"""
    tod = (index.hour * 60 + index.minute) / (24 * 60)   # 一天内的位置
    dow = index.dayofweek / 7                            # 星期几
    doy = (index.dayofyear - 1) / 366                    # 一年内的位置
    return pd.DataFrame(
        {
            "tod_sin": np.sin(2 * np.pi * tod),
            "tod_cos": np.cos(2 * np.pi * tod),
            "dow_sin": np.sin(2 * np.pi * dow),
            "dow_cos": np.cos(2 * np.pi * dow),
            "doy_sin": np.sin(2 * np.pi * doy),
            "doy_cos": np.cos(2 * np.pi * doy),
        },
        index=index,
    )


class WindowDataset(Dataset):
    """滑动窗口数据集。

    features: [N, C]，第 0 列为标准化后的功率，其余为时间特征
    target:   [N]，标准化后的功率（与 features 第 0 列相同）
    starts:   每个窗口的起始位置；输入 = [s, s+lookback)，标签 = [s+lookback, s+lookback+horizon)
    """

    def __init__(self, features: np.ndarray, target: np.ndarray,
                 lookback: int, horizon: int, starts: np.ndarray):
        self.features = features
        self.target = target
        self.lookback = lookback
        self.horizon = horizon
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        x = self.features[s: s + self.lookback]                                  # [L, C]
        y = self.target[s + self.lookback: s + self.lookback + self.horizon]     # [H]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_datasets(cfg: Config):
    df = load_dataframe(cfg)
    tfeat = make_time_features(df.index)

    # 标准化统计量只用训练段拟合，防止信息泄漏
    train_end_ts = pd.Timestamp(cfg.train_end)
    train_mask = df.index <= train_end_ts
    scaler = StandardScaler().fit(df.loc[train_mask, ["load"]].to_numpy(dtype=np.float32))
    load_scaled = scaler.transform(df[["load"]].to_numpy(dtype=np.float32))      # [N, 1]

    features = np.ascontiguousarray(
        np.concatenate([load_scaled, tfeat.to_numpy(dtype=np.float32)], axis=1)  # [N, C]
    )
    target = load_scaled[:, 0]                                                   # [N]

    # 按时间划分边界（验证/测试窗口的 lookback 允许回看上一段，标签不跨界）
    n = len(df)
    i_train_end = int(df.index.get_indexer([train_end_ts], method="pad")[0]) + 1
    i_val_end = int(df.index.get_indexer([pd.Timestamp(cfg.val_end)], method="pad")[0]) + 1
    lb, hz = cfg.lookback, cfg.horizon

    starts_train = np.arange(0, i_train_end - lb - hz + 1)
    starts_val = np.arange(i_train_end - lb, i_val_end - lb - hz + 1)
    starts_test = np.arange(i_val_end - lb, n - lb - hz + 1)

    train = WindowDataset(features, target, lb, hz, starts_train)
    val = WindowDataset(features, target, lb, hz, starts_val)
    test = WindowDataset(features, target, lb, hz, starts_test)
    print(f"[data] 窗口数  训练: {len(train)}  验证: {len(val)}  测试: {len(test)}")
    print(f"[data] 特征维度 C={features.shape[1]}（功率 + 6 个时间特征）")
    return train, val, test, scaler, df.index, i_val_end


def build_loaders(cfg: Config):
    train, val, test, scaler, index, i_val_end = build_datasets(cfg)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, generator=g)
    val_loader = DataLoader(val, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader, scaler, index, i_val_end
