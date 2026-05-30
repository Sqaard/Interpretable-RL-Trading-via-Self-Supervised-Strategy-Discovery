"""Policy-forward check for carefully transferring E9 rescorr groups to R-line Top-K.

This script evaluates group residual-quality modifiers without retraining. It
loads trained PPO models, recomputes policy anchors on the counterfactual
trajectory, and applies alternative controller configs. That is stronger than a
frozen-intent replay and is the minimum sanity check before promoting a variant
to Huawei packages.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from counterfactual_policy_forward_replay import (
    ARTIFACT_DIR,
    REPORT_DIR,
    CounterfactualCandidate,
    aggregate,
    process_zip,
    write_report,
)


def r6c_candidate(
    name: str,
    *,
    group_residual_quality: bool = False,
    buy_rank_weight: float = 0.60,
    sell_rank_weight: float = 0.85,
    buy_breadth_weight: float = 0.50,
    sell_breadth_weight: float = 0.50,
    residual_min_multiplier: float = 0.25,
    residual_max_multiplier: float = 2.50,
    top_k_buy: int = 8,
    top_k_sell: int = 12,
    rotation_budget: float = 0.0025,
    sell_residual_deterioration_weight: float = 0.75,
) -> CounterfactualCandidate:
    return CounterfactualCandidate(
        name=name,
        top_k_buy=top_k_buy,
        top_k_sell=top_k_sell,
        rotation_budget=rotation_budget,
        group_aware=True,
        default_group_cap=0.60,
        pressure_weight=1.00,
        capacity_weight=0.25,
        sell_overweight_weight=1.25,
        priority_floor=0.05,
        group_residual_quality=group_residual_quality,
        group_residual_buy_rank_weight=buy_rank_weight,
        group_residual_sell_rank_weight=sell_rank_weight,
        group_residual_buy_breadth_weight=buy_breadth_weight,
        group_residual_sell_breadth_weight=sell_breadth_weight,
        group_residual_min_multiplier=residual_min_multiplier,
        group_residual_max_multiplier=residual_max_multiplier,
        risk_aware_topk=True,
        buy_gate_min_confidence_rerisk=0.50,
        buy_gate_min_recovery_score=0.55,
        buy_gate_max_risk_stress=0.90,
        buy_gate_min_residual_breadth_excess_5d=-0.02,
        buy_gate_min_residual_breadth_excess_20d=-0.05,
        rotation_stress_gate_enabled=True,
        rotation_stress_start=0.55,
        rotation_stress_full=0.90,
        rotation_stress_min_scale=0.0,
        sell_risk_break_weight=1.0,
        sell_residual_deterioration_weight=sell_residual_deterioration_weight,
        sell_confidence_derisk_weight=0.25,
    )


def candidates() -> list[CounterfactualCandidate]:
    return [
        CounterfactualCandidate("pf_original"),
        r6c_candidate("pf_r6c_reference"),
        r6c_candidate(
            "pf_rescorr_groupquality_balanced_b8_s12",
            group_residual_quality=True,
            buy_rank_weight=0.60,
            sell_rank_weight=0.85,
            buy_breadth_weight=0.50,
            sell_breadth_weight=0.50,
        ),
        r6c_candidate(
            "pf_rescorr_groupquality_sellheavy_b8_s12",
            group_residual_quality=True,
            buy_rank_weight=0.45,
            sell_rank_weight=1.10,
            buy_breadth_weight=0.35,
            sell_breadth_weight=0.70,
        ),
        r6c_candidate(
            "pf_rescorr_groupquality_buyconfirm_b8_s12",
            group_residual_quality=True,
            buy_rank_weight=0.90,
            sell_rank_weight=0.85,
            buy_breadth_weight=0.75,
            sell_breadth_weight=0.50,
            residual_min_multiplier=0.20,
            residual_max_multiplier=2.25,
        ),
        r6c_candidate(
            "pf_rescorr_groupquality_balanced_b10_s12",
            group_residual_quality=True,
            top_k_buy=10,
            top_k_sell=12,
            buy_rank_weight=0.60,
            sell_rank_weight=0.85,
            buy_breadth_weight=0.50,
            sell_breadth_weight=0.50,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            str(ARTIFACT_DIR / "R6c_root_K20_stock_K5_PD_mild_slice_group_riskaware_top8_sell12_rotation_internaldays_v1"),
        ],
    )
    parser.add_argument(
        "--e9-rescorr-dir",
        default=str(ARTIFACT_DIR / "E9_hierarchical_discovered_rescorr_dirtree_fixedpd_v1"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPORT_DIR / "policy_forward_rescorr_groupquality_20260530"),
    )
    parser.add_argument("--max-zips", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = candidates()
    pd.DataFrame([asdict(candidate) for candidate in registry]).to_csv(
        output_dir / "counterfactual_candidate_registry.csv",
        index=False,
    )

    rows: list[dict[str, object]] = []
    zips_seen = 0
    for input_dir in args.input_dirs:
        for zip_path in sorted(Path(input_dir).glob("*.zip")):
            if args.max_zips and zips_seen >= args.max_zips:
                break
            rows.extend(
                process_zip(
                    zip_path,
                    e9_rescorr_dir=Path(args.e9_rescorr_dir),
                    output_dir=output_dir,
                    candidates=registry,
                )
            )
            zips_seen += 1
    fold_summary = pd.DataFrame(rows)
    fold_summary.to_csv(output_dir / "counterfactual_fold_summary.csv", index=False)
    agg = aggregate(fold_summary)
    agg.to_csv(output_dir / "counterfactual_aggregate_summary.csv", index=False)
    write_report(output_dir, fold_summary, agg)
    print(f"Wrote rescorr group-quality policy-forward outputs to {output_dir}")


if __name__ == "__main__":
    main()
