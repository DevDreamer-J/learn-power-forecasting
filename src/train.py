"""训练入口。

用法（在 python14 环境下，项目根目录执行）：
    python src/train.py --model patchtst
    python src/train.py --model dlinear

可选参数：--epochs --batch-size --lr --lookback --horizon --num-workers --cpu
输出（outputs/<model>/）：best.pt（最优权重）、metrics.json（测试指标）、pred_sample.png（预测曲线）
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # 无窗口环境直接出图
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import Config
from data import build_loaders
from models import DLinear, PatchTST

# 中文图例字体（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="电力负载预测训练")
    p.add_argument("--model", choices=["patchtst", "dlinear"], default="patchtst")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


def apply_args(cfg: Config, args: argparse.Namespace):
    """用命令行参数覆盖默认配置。"""
    cfg.model = args.model
    for arg_name, cfg_name in [("epochs", "epochs"), ("batch_size", "batch_size"),
                               ("lr", "lr"), ("lookback", "lookback"),
                               ("horizon", "horizon"), ("num_workers", "num_workers")]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)


def build_model(cfg: Config, n_channels: int) -> nn.Module:
    if cfg.model == "patchtst":
        return PatchTST(
            lookback=cfg.lookback, horizon=cfg.horizon, n_channels=n_channels,
            patch_len=cfg.patch_len, stride=cfg.patch_stride, d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_layers=cfg.n_layers, d_ff=cfg.d_ff,
            dropout=cfg.dropout)
    return DLinear(lookback=cfg.lookback, horizon=cfg.horizon, n_channels=n_channels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """还原到真实功率量纲（kW）后计算指标。"""
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mape = float((np.abs(y_true - y_pred) / np.abs(y_true)).mean() * 100)
    return {"MAE_kW": mae, "RMSE_kW": rmse, "MAPE_%": mape}


def evaluate(model: nn.Module, loader, criterion, scaler, device, cfg: Config):
    """返回 (标准化空间平均损失, 真实量纲 y_true [n, H], y_pred [n, H])。"""
    model.eval()
    total_loss, n_samples = 0.0, 0
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                pred = model(x)
                loss = criterion(pred.float(), y)
            total_loss += loss.item() * y.size(0)
            n_samples += y.size(0)
            ys.append(y.cpu().numpy())
            ps.append(pred.float().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_true = scaler.inverse(y_true)   # 还原到 kW
    y_pred = scaler.inverse(y_pred)
    return total_loss / n_samples, y_true, y_pred


def plot_sample(save_path: Path, y_true: np.ndarray, y_pred: np.ndarray,
                index, i_val_end: int, cfg: Config):
    """画出测试集第一个窗口起连续 3 天的真实/预测曲线。"""
    n_points = min(cfg.horizon * 3, y_true.shape[1])
    ts = index[i_val_end: i_val_end + n_points]
    plt.figure(figsize=(14, 5))
    plt.plot(ts, y_true[0, :n_points], label="实际值", linewidth=1.2, color="#1f77b4")
    plt.plot(ts, y_pred[0, :n_points], label="预测值", linewidth=1.2,
             alpha=0.85, color="#d62728")
    plt.title(f"{cfg.model.upper()} 负载预测 vs 实际（测试集前 3 天）")
    plt.xlabel("时间")
    plt.ylabel("总有功功率 (kW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = get_args()
    cfg = Config()
    apply_args(cfg, args)
    set_seed(cfg.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cfg.amp = cfg.amp and device.type == "cuda"

    train_loader, val_loader, test_loader, scaler, index, i_val_end = build_loaders(cfg)
    n_channels = train_loader.dataset.features.shape[1]

    model = build_model(cfg, n_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.model}  参数量 {n_params / 1e6:.2f}M  设备 {device}  AMP {cfg.amp}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
        pct_start=0.3, anneal_strategy="cos")
    amp_scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp)

    out_dir = cfg.out_dir / cfg.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    # ---------- 训练 + 验证（早停）----------
    best_val, wait = float("inf"), 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp):
                pred = model(x)
                loss = criterion(pred.float(), y)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            scheduler.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)

        val_loss, _, _ = evaluate(model, val_loader, criterion, scaler, device, cfg)
        lr_now = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch:03d}] train_mse={running / seen:.4f}  "
              f"val_mse={val_loss:.4f}  lr={lr_now:.2e}  {time.time() - t0:.1f}s")

        if val_loss < best_val - 1e-5:
            best_val, wait = val_loss, 0
            torch.save({"model": model.state_dict(),
                        "config": {k: str(v) for k, v in vars(cfg).items()}}, ckpt_path)
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[early stop] 验证集 {cfg.patience} 轮未提升，提前结束训练")
                break

    # ---------- 测试 ----------
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["model"])
    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, scaler, device, cfg)
    metrics = compute_metrics(y_true, y_pred)
    print(f"[test] 标准化 MSE={test_loss:.4f}  "
          + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    (out_dir / "metrics.json").write_text(
        json.dumps({"model": cfg.model, "val_mse": best_val,
                    "test_standardized_mse": test_loss, **metrics},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    plot_sample(out_dir / "pred_sample.png", y_true, y_pred, index, i_val_end, cfg)
    print(f"[done] 指标与曲线已保存到 {out_dir}")


if __name__ == "__main__":
    main()
