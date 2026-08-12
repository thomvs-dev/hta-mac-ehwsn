"""Render tables, figures, and a manuscript-ready results capsule from final evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POLICY_LABELS = {
    "learned_listwise_residual": "HTA-MAC residual",
    "analytic_teacher": "Analytic teacher",
    "energy_proportional_tuned": "Energy-proportional",
    "s2a2mac_adapted": "S2A2MAC-adapted",
    "ffss_adapted": "FFSS-adapted",
}
METRIC_LABELS = {
    "delivery_ratio": "Delivery ratio",
    "stale_ratio": "Stale ratio",
    "episode_service_fairness": "Service fairness",
    "fnd_free_steps": "FND (rounds)",
    "global_packets_per_j": "Packets/J",
}


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_ci(values, rng, resamples=20000):
    values = np.asarray(values, dtype=np.float64)
    sample = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return np.quantile(sample, [0.025, 0.975])


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    if not payload.get("integrity_pass"):
        raise RuntimeError("refusing to render failed final evidence")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260813)
    metrics = list(METRIC_LABELS)
    policies = [policy for policy in POLICY_LABELS if policy in payload["aggregates"]]

    aggregate_rows = []
    for policy in policies:
        seed_rows = payload["seed_means"][policy]
        row = {
            "policy": policy,
            "label": POLICY_LABELS[policy],
            "joint_qos_passes": payload["aggregates"][policy]["joint_qos_pass_count"],
            "trials": payload["aggregates"][policy]["rows"],
        }
        for metric in metrics:
            values = [entry[metric] for entry in seed_rows.values()]
            ci = seed_ci(values, rng)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"ci_low_{metric}"] = float(ci[0])
            row[f"ci_high_{metric}"] = float(ci[1])
        aggregate_rows.append(row)
    aggregate_fields = list(aggregate_rows[0])
    write_csv(output / "table_final_aggregate_metrics.csv", aggregate_fields, aggregate_rows)

    paired_rows = []
    for comparator, comparisons in payload["paired_inference_holm_family"].items():
        for metric, result in comparisons.items():
            paired_rows.append({
                "comparator": comparator,
                "metric": metric,
                "direction": result["direction"],
                "mean_hta_minus_comparator": result["mean_raw_difference"],
                "ci_low": result["bootstrap_95_ci_raw_difference"][0],
                "ci_high": result["bootstrap_95_ci_raw_difference"][1],
                "wilcoxon_p": result["wilcoxon_p_value_unadjusted"],
                "holm_p": result["wilcoxon_p_value_holm"],
                "holm_significant": result["reject_holm_0_05"],
                "paired_dz_favorable": result["paired_cohens_dz_favorable"],
                "rank_biserial_favorable": result["paired_rank_biserial_favorable"],
                **result["wins_ties_losses_favorable"],
            })
    write_csv(output / "table_final_paired_inference.csv", list(paired_rows[0]), paired_rows)

    colors = ["#2463EB", "#94A3B8", "#F59E0B", "#DC2626", "#16A34A"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()
    for axis, metric in zip(axes, metrics):
        means, lower, upper = [], [], []
        for row in aggregate_rows:
            mean = row[f"mean_{metric}"]
            means.append(mean)
            lower.append(mean - row[f"ci_low_{metric}"])
            upper.append(row[f"ci_high_{metric}"] - mean)
        x = np.arange(len(policies))
        axis.bar(x, means, color=colors, alpha=0.88)
        axis.errorbar(x, means, yerr=np.vstack((lower, upper)), fmt="none", ecolor="#111827", capsize=3)
        axis.set_title(METRIC_LABELS[metric], fontweight="bold")
        axis.set_xticks(x, [POLICY_LABELS[p].replace("-", "‑") for p in policies], rotation=25, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    axis = axes[-1]
    passes = [payload["aggregates"][policy]["joint_qos_pass_count"] for policy in policies]
    axis.bar(np.arange(len(policies)), passes, color=colors, alpha=0.88)
    axis.axhline(90, color="#111827", linestyle="--", linewidth=1, label="predeclared HTA gate")
    axis.set_title("Joint QoS passes (of 100)", fontweight="bold")
    axis.set_xticks(np.arange(len(policies)), [POLICY_LABELS[p].replace("-", "‑") for p in policies], rotation=25, ha="right", fontsize=8)
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Final matched-simulator evaluation: 20 independent seeds × 5 nested schedules", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output / "figure_final_policy_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    comparators = list(payload["paired_inference_holm_family"])
    figure, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    for axis, metric in zip(axes, metrics):
        differences, lower, upper = [], [], []
        for comparator in comparators:
            result = payload["paired_inference_holm_family"][comparator][metric]
            mean = result["mean_raw_difference"]
            differences.append(mean)
            lower.append(mean - result["bootstrap_95_ci_raw_difference"][0])
            upper.append(result["bootstrap_95_ci_raw_difference"][1] - mean)
        y = np.arange(len(comparators))
        axis.errorbar(differences, y, xerr=np.vstack((lower, upper)), fmt="o", color="#2463EB", ecolor="#475569", capsize=3)
        axis.axvline(0, color="#111827", linewidth=1)
        axis.set_yticks(y, [POLICY_LABELS[c] for c in comparators] if metric == metrics[0] else [])
        axis.set_title(METRIC_LABELS[metric], fontweight="bold", fontsize=10)
        axis.set_xlabel("HTA - comparator", fontsize=8)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Seed-paired mean differences with 95% bootstrap CIs (n=20 seeds)", fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output / "figure_final_paired_differences.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    aggregate_map = {row["policy"]: row for row in aggregate_rows}
    hta = aggregate_map["learned_listwise_residual"]
    energy = payload["paired_inference_holm_family"]["energy_proportional_tuned"]
    report = f"""# Final matched-simulator results capsule

## Design

The frozen primary idle-listening-on simulator was evaluated on 20 previously unused seeds (3700-3719), with five target-rank schedules nested within every seed. The independent inferential unit was the seed (n=20), not the 100 correlated seed/rank rows. Each policy used the same schedules, 3,000-round horizon, frozen HEART-CH schedule, and MAC-only action scope. No selection or retuning occurred after the cohort was opened.

The prespecified family contained five metrics against three fixed comparators (15 paired hypotheses). Two-sided Wilcoxon signed-rank tests used seed-level means, 95% paired-bootstrap confidence intervals used 20,000 resamples, and p-values were Holm-corrected across the full family.

## Main result

HTA-MAC passed the joint delivery/staleness/fairness constraints in **{hta['joint_qos_passes']}/100** matched trials. The tuned energy-proportional baseline passed {aggregate_map['energy_proportional_tuned']['joint_qos_passes']}/100, FFSS-adapted {aggregate_map['ffss_adapted']['joint_qos_passes']}/100, and S2A2MAC-adapted {aggregate_map['s2a2mac_adapted']['joint_qos_passes']}/100.

Against tuned energy-proportional, HTA-MAC had slightly lower delivery (paired difference {fmt(energy['delivery_ratio']['mean_raw_difference'])}, 95% CI [{fmt(energy['delivery_ratio']['bootstrap_95_ci_raw_difference'][0])}, {fmt(energy['delivery_ratio']['bootstrap_95_ci_raw_difference'][1])}], Holm p={energy['delivery_ratio']['wilcoxon_p_value_holm']:.4g}) but lower stale ratio ({fmt(energy['stale_ratio']['mean_raw_difference'])}, 95% CI [{fmt(energy['stale_ratio']['bootstrap_95_ci_raw_difference'][0])}, {fmt(energy['stale_ratio']['bootstrap_95_ci_raw_difference'][1])}], Holm p={energy['stale_ratio']['wilcoxon_p_value_holm']:.4g}) and higher service fairness (+{fmt(energy['episode_service_fairness']['mean_raw_difference'])}, 95% CI [{fmt(energy['episode_service_fairness']['bootstrap_95_ci_raw_difference'][0])}, {fmt(energy['episode_service_fairness']['bootstrap_95_ci_raw_difference'][1])}], Holm p={energy['episode_service_fairness']['wilcoxon_p_value_holm']:.4g}).

The lifetime result is a genuine trade-off, not dominance: HTA-MAC reached FND {abs(energy['fnd_free_steps']['mean_raw_difference']):.2f} rounds earlier than energy-proportional (95% CI [{energy['fnd_free_steps']['bootstrap_95_ci_raw_difference'][0]:.2f}, {energy['fnd_free_steps']['bootstrap_95_ci_raw_difference'][1]:.2f}], Holm p={energy['fnd_free_steps']['wilcoxon_p_value_holm']:.4g}) and delivered {abs(energy['global_packets_per_j']['mean_raw_difference']):.3f} fewer packets/J (Holm p={energy['global_packets_per_j']['wilcoxon_p_value_holm']:.4g}). The defensible contribution is therefore reliable joint-QoS control and learned approximation of the analytic projection, not universal lifetime or efficiency superiority.

## Claim boundary

These are matched-simulator comparisons. S2A2MAC and FFSS are documented structural adaptations because their original simulators and unpublished parameters are not interchangeable with this environment. The results must not be presented as direct reproduction of, or numerical superiority over, the source papers.
"""
    (output / "FINAL_RESULTS_CAPSULE.md").write_text(report, encoding="utf-8")
    print(f"WROTE={output}")


if __name__ == "__main__":
    main()
