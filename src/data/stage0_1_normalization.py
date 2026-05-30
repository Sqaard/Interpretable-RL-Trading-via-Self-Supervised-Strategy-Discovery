"""Fold-local feature normalization for Stage 0.1 PPO training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
KEY_COLUMNS = {"date", "tic", "close"}
EPS = 1e-12


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEY_COLUMNS]


def causal_impute(df: pd.DataFrame, feature_columns: list[str], train_mask: pd.Series) -> pd.DataFrame:
    """Impute without using post-train summary statistics."""
    out = df.copy()
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(["tic", "date"]).reset_index(drop=True)
    train_mask = out["date"].between(df.loc[train_mask, "date"].min(), df.loc[train_mask, "date"].max())
    out[feature_columns] = out.groupby("tic", sort=False)[feature_columns].ffill()
    train_medians = out.loc[train_mask, feature_columns].median(numeric_only=True)
    out[feature_columns] = out[feature_columns].fillna(train_medians).fillna(0.0)
    return out.sort_values(["date", "tic"]).reset_index(drop=True)


def fit_transform_stats(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_mask: pd.Series,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    rows = []
    for col in feature_columns:
        train_values = df.loc[train_mask, col].astype(float)
        lower = float(train_values.quantile(lower_quantile))
        upper = float(train_values.quantile(upper_quantile))
        clipped = train_values.clip(lower, upper)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0))
        if not np.isfinite(std) or std < EPS:
            std = 1.0
        rows.append(
            {
                "feature": col,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_value_train": lower,
                "upper_value_train": upper,
                "mean_after_clip_train": mean,
                "std_after_clip_train": std,
            }
        )
    return pd.DataFrame(rows)


def apply_transform(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for row in stats.itertuples(index=False):
        clipped = out[row.feature].astype(float).clip(row.lower_value_train, row.upper_value_train)
        out[row.feature] = (clipped - row.mean_after_clip_train) / row.std_after_clip_train
    return out


def diagnostics(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "feature_count": int(len(feature_columns)),
                "ticker_count": int(df["tic"].nunique()),
                "date_count": int(df["date"].nunique()),
                "start_date": str(pd.to_datetime(df["date"]).min().date()),
                "end_date": str(pd.to_datetime(df["date"]).max().date()),
                "missing_feature_cells": int(df[feature_columns].isna().sum().sum()),
                "inf_feature_cells": int(np.isinf(df[feature_columns].to_numpy(dtype=float)).sum()),
                "duplicate_date_tic_rows": int(df.duplicated(["date", "tic"]).sum()),
            }
        ]
    )


def prepare_fold_scaled_features(
    *,
    raw_csv: str | Path,
    out_dir: str | Path,
    fold_id: str,
    train_start: str,
    train_end: str,
    validation_end: str,
    feature_subset: list[str] | None = None,
    feature_subset_name: str | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    force: bool = False,
) -> dict[str, Any]:
    """Create a fold-specific model-ready CSV using train-only scaling stats."""
    raw_path = Path(raw_csv)
    fold_dir = Path(out_dir) / fold_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_ready_path = fold_dir / "model_ready.csv"
    stats_path = fold_dir / "transform_stats.csv"
    diagnostics_path = fold_dir / "diagnostics.csv"
    manifest_path = fold_dir / "manifest.json"

    if (
        not force
        and model_ready_path.exists()
        and stats_path.exists()
        and diagnostics_path.exists()
        and manifest_path.exists()
        and model_ready_path.stat().st_mtime >= raw_path.stat().st_mtime
    ):
        return {
            "model_ready_csv": model_ready_path,
            "transform_stats_csv": stats_path,
            "diagnostics_csv": diagnostics_path,
            "manifest_json": manifest_path,
            "status": "cached",
            "feature_subset_name": feature_subset_name,
        }

    df = pd.read_csv(raw_path)
    required = {"date", "tic", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{raw_path} is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])

    start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    needed_mask = df["date"].between(start_ts, validation_end_ts)
    fold_df = df.loc[needed_mask].copy()
    if fold_df.empty:
        raise ValueError(f"No rows in {raw_path} for fold {fold_id}.")

    if feature_subset is not None:
        missing_subset = [col for col in feature_subset if col not in fold_df.columns]
        if missing_subset:
            raise ValueError(
                f"Feature subset {feature_subset_name or '<unnamed>'} contains missing columns: "
                f"{missing_subset}"
            )
        fold_df = fold_df[["date", "tic", "close", *feature_subset]].copy()
        feature_columns = list(feature_subset)
    else:
        feature_columns = get_feature_columns(fold_df)
    train_mask = fold_df["date"].between(start_ts, train_end_ts)
    if not train_mask.any():
        raise ValueError(f"No train rows in {raw_path} for fold {fold_id}.")

    fold_df = causal_impute(fold_df, feature_columns, train_mask)
    train_mask = fold_df["date"].between(start_ts, train_end_ts)
    stats = fit_transform_stats(
        fold_df,
        feature_columns,
        train_mask,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    model_ready = apply_transform(fold_df, stats)
    model_ready = model_ready.sort_values(["date", "tic"]).reset_index(drop=True)

    model_ready.to_csv(model_ready_path, index=False)
    stats.to_csv(stats_path, index=False)
    diagnostics(model_ready, feature_columns).to_csv(diagnostics_path, index=False)

    manifest = {
        "fold_id": fold_id,
        "raw_csv": rel(raw_path),
        "model_ready_csv": rel(model_ready_path),
        "transform_stats_csv": rel(stats_path),
        "diagnostics_csv": rel(diagnostics_path),
        "train_start": train_start,
        "train_end": train_end,
        "validation_end": validation_end,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "feature_subset_name": feature_subset_name,
        "normalization": "fold-train-only winsorization followed by fold-train-only z-score",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "model_ready_csv": model_ready_path,
        "transform_stats_csv": stats_path,
        "diagnostics_csv": diagnostics_path,
        "manifest_json": manifest_path,
        "status": "created",
        "feature_subset_name": feature_subset_name,
    }
