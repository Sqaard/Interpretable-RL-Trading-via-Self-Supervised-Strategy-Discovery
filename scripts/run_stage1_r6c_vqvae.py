"""Train a neural windowed VQ-VAE Stage 1 codebook for the R6c teacher.

This is the neural counterpart to ``run_stage1_r6c_vq.py``.  It consumes the
already extracted R6c Stage 1 hidden-state package and behavior logs, trains a
small PyTorch VQ-VAE over centered hidden-state windows, and writes Joseph-
compatible Stage 1 artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT_DIR = (
    ROOT
    / "artifacts"
    / "stage1"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_stage1"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "stage1"
    / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_stage1_vqvae"
)


@dataclass
class TrainResult:
    k: int
    best_epoch: int
    best_val_recon: float
    recon_mse_normalized: float
    var_h: float
    recon_over_var: float
    utilization: float
    perplexity: float
    median_run_length: float
    cross_fold_nmi: float
    codes: np.ndarray
    valid: np.ndarray
    model_state_path: str


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, latent_dim: int, commitment_beta: float) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.latent_dim = int(latent_dim)
        self.commitment_beta = float(commitment_beta)
        self.embedding = nn.Embedding(self.num_codes, self.latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_codes, 1.0 / self.num_codes)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = (
            torch.sum(z_e**2, dim=1, keepdim=True)
            - 2.0 * z_e @ self.embedding.weight.t()
            + torch.sum(self.embedding.weight**2, dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        codebook_loss = torch.mean((z_q - z_e.detach()) ** 2)
        commitment_loss = torch.mean((z_e - z_q.detach()) ** 2)
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss
        z_st = z_e + (z_q - z_e).detach()
        return z_st, indices, vq_loss

    def encode(self, z_e: torch.Tensor) -> torch.Tensor:
        distances = (
            torch.sum(z_e**2, dim=1, keepdim=True)
            - 2.0 * z_e @ self.embedding.weight.t()
            + torch.sum(self.embedding.weight**2, dim=1)
        )
        return torch.argmin(distances, dim=1)


class WindowVQVAE(nn.Module):
    def __init__(self, input_dim: int, num_codes: int, latent_dim: int, hidden_dim: int, commitment_beta: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.vq = VectorQuantizer(num_codes, latent_dim, commitment_beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, indices, vq_loss

    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        return self.vq.encode(z_e)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def centered_windows(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window % 2 != 1:
        raise ValueError("window_length must be odd")
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty((len(x), window * x.shape[1]), dtype=np.float32)
    for i in range(len(x)):
        out[i] = padded[i : i + window].reshape(-1)
    valid = np.ones(len(x), dtype=bool)
    if pad:
        valid[:pad] = False
        valid[-pad:] = False
    return out, valid


def run_lengths(codes: np.ndarray, valid: np.ndarray) -> list[int]:
    seq = codes[valid]
    if len(seq) == 0:
        return []
    runs: list[int] = []
    current = int(seq[0])
    length = 1
    for value in seq[1:]:
        ivalue = int(value)
        if ivalue == current:
            length += 1
        else:
            runs.append(length)
            current = ivalue
            length = 1
    runs.append(length)
    return runs


def code_stats(codes: np.ndarray, valid: np.ndarray, k: int) -> tuple[float, float, np.ndarray]:
    counts = np.bincount(codes[valid], minlength=k).astype(float)
    probs = counts / max(float(counts.sum()), 1.0)
    active = counts > 0
    nonzero = probs[probs > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) if len(nonzero) else 0.0
    return float(active.sum() / k), float(np.exp(entropy)), counts


def encode_numpy(model: WindowVQVAE, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    codes: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float()
            codes.append(model.encode_codes(batch).cpu().numpy().astype(np.int32))
    return np.concatenate(codes)


def reconstruct_mse(model: WindowVQVAE, x: np.ndarray, batch_size: int = 512) -> float:
    model.eval()
    losses: list[float] = []
    counts: list[int] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float()
            recon, _, _ = model(batch)
            loss = torch.mean((recon - batch) ** 2).item()
            losses.append(loss)
            counts.append(len(batch))
    return float(np.average(losses, weights=counts)) if losses else math.nan


def summarize_codes(frame: pd.DataFrame, *, k: int) -> pd.DataFrame:
    valid_frame = frame.loc[frame["valid"]].copy()
    denom = max(len(valid_frame), 1)
    rows: list[dict[str, Any]] = []
    for code in range(k):
        sub = valid_frame.loc[valid_frame["code_id"] == code]
        rows.append(
            {
                "code_id": code,
                "n": int(len(sub)),
                "freq": float(len(sub) / denom),
                "mean_return": float(pd.to_numeric(sub.get("net_return"), errors="coerce").mean()) if len(sub) else np.nan,
                "median_return": float(pd.to_numeric(sub.get("net_return"), errors="coerce").median()) if len(sub) else np.nan,
                "mean_vix": float(pd.to_numeric(sub.get("market_feature_VIX"), errors="coerce").mean()) if len(sub) else np.nan,
                "mean_regime_p1": float(pd.to_numeric(sub.get("market_feature_Regime_1_Prob"), errors="coerce").mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def aggregate_behavior_by_code(frame: pd.DataFrame, *, k: int) -> pd.DataFrame:
    useful_patterns = (
        "q_",
        "cash_",
        "risk_stress",
        "recovery_score",
        "confidence_",
        "root_",
        "trade_",
        "turnover",
        "drawdown",
        "concentration",
        "topk_",
        "incremental_topk_",
        "market_feature_",
        "net_return",
        "gross_return",
        "reward",
    )
    numeric_cols = [
        col
        for col in frame.columns
        if col not in {"code_id", "date", "next_date", "fold", "split", "source_zip"}
        and any(pattern in col for pattern in useful_patterns)
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    rows: list[dict[str, Any]] = []
    valid_frame = frame.loc[frame["valid"]].copy()
    for code in range(k):
        sub = valid_frame.loc[valid_frame["code_id"] == code]
        row: dict[str, Any] = {"code_id": code, "n": int(len(sub))}
        for col in numeric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            if values.notna().any():
                row[f"mean_{col}"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def cross_fold_nmi_same_encoder(
    model: WindowVQVAE,
    input_dir: Path,
    primary_dates: np.ndarray,
    primary_codes: np.ndarray,
    primary_valid: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    window_length: int,
) -> float:
    primary_map = {
        str(date): int(code)
        for date, code, valid in zip(primary_dates, primary_codes, primary_valid)
        if bool(valid)
    }
    scores: list[float] = []
    for hidden_path in sorted((input_dir / "per_fold").glob("fold_*_*_hidden.npz")):
        data = np.load(hidden_path, allow_pickle=True)
        dates = np.asarray(data["dates"]).astype(str)
        hidden = np.asarray(data["hidden"], dtype=np.float32)
        z = ((hidden - mean) / std).astype(np.float32)
        windows, valid = centered_windows(z, window_length)
        codes = encode_numpy(model, windows)
        a: list[int] = []
        b: list[int] = []
        for date, code, is_valid in zip(dates, codes, valid):
            if not bool(is_valid) or str(date) not in primary_map:
                continue
            a.append(primary_map[str(date)])
            b.append(int(code))
        if len(set(a)) > 1 and len(set(b)) > 1:
            scores.append(float(normalized_mutual_info_score(a, b)))
    return float(np.mean(scores)) if scores else math.nan


def train_one_k(
    *,
    k: int,
    windows: np.ndarray,
    valid: np.ndarray,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    hidden_dim: int,
    commitment_beta: float,
) -> tuple[WindowVQVAE, int, float]:
    set_seed(seed + k)
    valid_idx = np.flatnonzero(valid)
    train_cut = int(len(valid_idx) * 0.80)
    train_idx = valid_idx[:train_cut]
    val_idx = valid_idx[train_cut:]
    x_train = torch.from_numpy(windows[train_idx]).float()
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=False)

    model = WindowVQVAE(
        input_dim=windows.shape[1],
        num_codes=k,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        commitment_beta=commitment_beta,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_epoch = -1
    best_val = math.inf
    best_state: dict[str, Any] | None = None
    patience = 15
    bad_epochs = 0
    val_x = windows[val_idx]

    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in loader:
            opt.zero_grad(set_to_none=True)
            recon, _, vq_loss = model(batch)
            recon_loss = torch.mean((recon - batch) ** 2)
            loss = recon_loss + vq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        val_recon = reconstruct_mse(model, val_x)
        if val_recon + 1e-7 < best_val:
            best_val = val_recon
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = out_dir / "vqvae_models" / f"vqvae_K{k}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "k": k,
            "input_dim": windows.shape[1],
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "commitment_beta": commitment_beta,
            "best_epoch": best_epoch,
            "best_val_recon": best_val,
        },
        model_path,
    )
    return model, best_epoch, best_val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selected-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--window-length", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--commitment-beta", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hidden_npz = np.load(args.input_dir / "r6c_stage1_hidden_state_package.npz", allow_pickle=True)
    dates = np.asarray(hidden_npz["train_dates"]).astype(str)
    hidden = np.asarray(hidden_npz["train_hidden"], dtype=np.float32)
    mean, std = zscore_fit(hidden)
    z = ((hidden - mean) / std).astype(np.float32)
    windows, valid = centered_windows(z, args.window_length)
    behavior = pd.read_parquet(args.input_dir / "behavior_log_daily.parquet").copy()
    if len(behavior) != len(hidden):
        raise ValueError(f"Behavior rows {len(behavior)} != hidden rows {len(hidden)}")

    results: list[TrainResult] = []
    selected_model: WindowVQVAE | None = None
    selected_result: TrainResult | None = None
    for k in args.candidate_k:
        print(f"Training neural VQ-VAE K={k}", flush=True)
        model, best_epoch, best_val = train_one_k(
            k=k,
            windows=windows,
            valid=valid,
            out_dir=args.out_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            commitment_beta=args.commitment_beta,
        )
        codes = encode_numpy(model, windows)
        all_recon = reconstruct_mse(model, windows[valid])
        var_h = float(np.var(windows[valid]))
        utilization, perplexity, _ = code_stats(codes, valid, k)
        runs = run_lengths(codes, valid)
        cross_nmi = cross_fold_nmi_same_encoder(
            model,
            args.input_dir,
            dates,
            codes,
            valid,
            mean=mean,
            std=std,
            window_length=args.window_length,
        )
        result = TrainResult(
            k=k,
            best_epoch=best_epoch,
            best_val_recon=best_val,
            recon_mse_normalized=all_recon,
            var_h=var_h,
            recon_over_var=all_recon / var_h if var_h > 0 else math.nan,
            utilization=utilization,
            perplexity=perplexity,
            median_run_length=float(np.median(runs)) if runs else math.nan,
            cross_fold_nmi=cross_nmi,
            codes=codes.astype(np.int32),
            valid=valid.copy(),
            model_state_path=str(args.out_dir / "vqvae_models" / f"vqvae_K{k}.pt"),
        )
        results.append(result)
        if k == args.selected_k:
            selected_model = model
            selected_result = result

    if selected_result is None or selected_model is None:
        raise ValueError(f"selected_k={args.selected_k} was not trained")

    metrics = pd.DataFrame(
        [
            {
                "K": r.k,
                "best_val_recon": r.best_val_recon,
                "best_epoch": r.best_epoch,
                "utilization": r.utilization,
                "perplexity": r.perplexity,
                "recon_mse_normalized": r.recon_mse_normalized,
                "var_h": r.var_h,
                "recon_over_var": r.recon_over_var,
                "median_run_length": r.median_run_length,
                "cross_fold_nmi": r.cross_fold_nmi,
                "source": "vqvae_r6c",
            }
            for r in results
        ]
    )
    metrics.to_csv(args.out_dir / "stage1_metrics.csv", index=False)
    metrics.loc[metrics["K"].eq(args.selected_k)].to_csv(args.out_dir / "stage1_metrics_selected_k.csv", index=False)

    coded = behavior.copy()
    coded["code_id"] = selected_result.codes
    coded["valid"] = selected_result.valid
    train_codes = coded[["date", "code_id", "valid"]].copy()
    train_codes.to_parquet(args.out_dir / "train_codes.parquet", index=False)
    train_codes.to_csv(args.out_dir / "train_codes.csv", index=False)
    summarize_codes(coded, k=args.selected_k).to_csv(args.out_dir / "code_summary.csv", index=False)
    coded.to_parquet(args.out_dir / "behavior_log_daily.parquet", index=False)
    coded.to_csv(args.out_dir / "behavior_log_daily.csv", index=False)
    aggregate_behavior_by_code(coded, k=args.selected_k).to_csv(args.out_dir / "code_behavior_summary.csv", index=False)

    np.savez_compressed(
        args.out_dir / "r6c_stage1_hidden_state_package_vqvae.npz",
        model_id=np.asarray(["R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_vqvae"], dtype="U128"),
        train_dates=dates,
        train_hidden=hidden,
        train_code_id=selected_result.codes.astype(np.int32),
        train_code_valid=selected_result.valid.astype(bool),
        zscore_mean=mean.astype(np.float32),
        zscore_std=std.astype(np.float32),
        window_length=np.asarray([args.window_length], dtype=np.int32),
        selected_k=np.asarray([args.selected_k], dtype=np.int32),
        train_net_return=pd.to_numeric(coded.get("net_return"), errors="coerce").to_numpy(dtype=np.float32),
        train_cash_target=pd.to_numeric(coded.get("cash_target"), errors="coerce").to_numpy(dtype=np.float32),
        train_risk_stress=pd.to_numeric(coded.get("risk_stress"), errors="coerce").to_numpy(dtype=np.float32),
        train_recovery_score=pd.to_numeric(coded.get("recovery_score"), errors="coerce").to_numpy(dtype=np.float32),
        train_confidence_derisk=pd.to_numeric(coded.get("confidence_derisk"), errors="coerce").to_numpy(dtype=np.float32),
        train_confidence_rerisk=pd.to_numeric(coded.get("confidence_rerisk"), errors="coerce").to_numpy(dtype=np.float32),
    )

    manifest = {
        "model": "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1",
        "method": "neural_windowed_vqvae",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.out_dir),
        "selected_k": args.selected_k,
        "candidate_k": args.candidate_k,
        "window_length": args.window_length,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "commitment_beta": args.commitment_beta,
        "epochs": args.epochs,
        "metrics": metrics.to_dict(orient="records"),
        "alignment_note": (
            "Input hidden states come from policy-call replay. For K-window macro-steps, "
            "one hidden state is broadcast to the internal daily rows controlled by that policy call."
        ),
    }
    (args.out_dir / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / "README_STAGE1_R6C_VQVAE.md").write_text(
        f"""# Neural VQ-VAE Stage 1 for R6c

This is the full neural Stage 1 variant for the selected R6c teacher.

It trains a PyTorch windowed VQ-VAE over centered hidden-state windows and
writes Joseph-compatible files:

- `train_codes.parquet`
- `code_summary.csv`
- `stage1_metrics.csv`

It also preserves the R6c behavior interpretability handoff:

- `behavior_log_daily.parquet`
- `code_behavior_summary.csv`
- `r6c_stage1_hidden_state_package_vqvae.npz`

Selected K: `{args.selected_k}`
Window length: `{args.window_length}`
Source: `vqvae_r6c`

Use `valid == True` before return/calibration analysis.
""",
        encoding="utf-8",
    )
    print(f"Neural VQ-VAE Stage 1 package written to {args.out_dir}")


if __name__ == "__main__":
    main()
