"""Validate the frozen HEART-CH foundation before any HTA-MAC implementation.

This validator is intentionally read-only with respect to the upstream
HEART-CH repository. It records raw per-trial metrics and reproducibility
metadata under HTA-MAC's outputs/logs directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
ASSET_MANIFEST_PATH = PROJECT_ROOT / "core" / "frozen_assets.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    """Load the deliberately simple Phase 0 YAML subset without PyYAML."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: unsupported YAML line")
        key, raw_value = (part.strip() for part in stripped.split(":", 1))
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if not raw_value:
            value: object = {}
            parent[key] = value
            stack.append((indent, value))
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parent[key] = value
    return root


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_base_config(config: dict) -> list[str]:
    expected = {
        ("network", "num_nodes"): 100,
        ("network", "field_size_m"): [100.0, 100.0],
        ("network", "bs_position_m"): [50.0, 175.0],
        ("network", "initial_energy_j"): 0.5,
        ("network", "packet_bits"): 4000,
        ("network", "control_packet_bits"): 100,
        ("network", "ch_ratio"): 0.05,
        ("network", "tx_range_m"): 50.0,
        ("radio", "e_elec_j_per_bit"): 50e-9,
        ("radio", "eps_fs_j_per_bit_m2"): 10e-12,
        ("radio", "eps_mp_j_per_bit_m4"): 0.0013e-12,
        ("radio", "e_da_j_per_bit"): 5e-9,
        ("model", "temporal_window"): 10,
        ("model", "node_features"): 31,
        ("mobility", "mobile_fraction"): 0.20,
        ("mobility", "speed_m_per_round"): [1.0, 5.0],
        ("harvesting", "solar"): {"states": 8, "rectification_scale": 0.01},
        ("harvesting", "thermal"): {"states": 4, "rectification_scale": 0.005},
    }
    failures = []
    for keys, wanted in expected.items():
        actual = config
        for key in keys:
            actual = actual[key]
        if actual != wanted:
            failures.append(f"{'.'.join(keys)}={actual!r}, expected {wanted!r}")

    radio = config["radio"]
    calculated_d0 = np.sqrt(
        radio["eps_fs_j_per_bit_m2"] / radio["eps_mp_j_per_bit_m4"]
    )
    if not np.isclose(calculated_d0, radio["d0_m"], rtol=0.0, atol=1e-12):
        failures.append(
            f"radio.d0_m={radio['d0_m']}, calculated {calculated_d0}"
        )
    return failures


def inspect_hmm_artifacts(upstream: Path, manifest: dict) -> tuple[dict, list[str]]:
    failures = []
    solar_spec = manifest["solar_hmm"]
    solar_path = upstream / solar_spec["path"]
    solar = loadmat(solar_path)
    solar_values = {
        key: np.asarray(solar[key]).tolist() for key in solar_spec["required_keys"]
        if key in solar
    }
    missing_solar = sorted(set(solar_spec["required_keys"]) - set(solar_values))
    if missing_solar:
        failures.append(f"solar HMM keys missing: {missing_solar}")

    thermal_spec = manifest["thermal_hmm"]
    thermal_path = thermal_spec.get("path")
    if not thermal_path:
        failures.append(
            "trained thermal HMM artifact missing; upstream synthesizes thermal parameters"
        )
        thermal_values = None
    else:
        thermal_file = upstream / thermal_path
        thermal_values = loadmat(thermal_file)

    return {
        "solar": solar_values,
        "thermal": thermal_values,
    }, failures


def make_upstream_components(upstream: Path, max_rounds: int):
    sys.path.insert(0, str(upstream))
    import config as cfg
    from env.wsn_env import EH_WSN_Env
    from train import build_agent, load_stage1_params

    params = load_stage1_params(str(upstream / "outputs" / "stage1_params.mat"))
    env = EH_WSN_Env(
        num_nodes=cfg.NUM_NODES,
        area_size=cfg.AREA_SIZE,
        bs_position=cfg.BS_POSITION.tolist(),
        tx_range=cfg.TX_RANGE,
        ch_ratio=cfg.CH_RATIO,
        initial_energy=cfg.INITIAL_ENERGY,
        e_elec=cfg.E_ELEC,
        e_fs=cfg.E_FS,
        e_mp=cfg.E_MP,
        e_da=cfg.E_DA,
        packet_size=cfg.PACKET_SIZE,
        control_packet_size=cfg.CONTROL_PACKET_SIZE,
        max_rounds=max_rounds,
        num_features=cfg.NUM_NODE_FEATURES,
        temporal_window=cfg.TEMPORAL_WINDOW,
        hmm_A=params["hmm_A"],
        hmm_mu=params["hmm_mu"],
        hmm_sigma2=params["hmm_sigma2"],
        hmm_pi0=params["hmm_pi0"],
    )
    agent, _ = build_agent(device="cpu", mode=cfg.AGENT_MODE)
    checkpoint_path = upstream / "outputs" / "checkpoints" / "model_v91_throughput.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.online_net.load_state_dict(checkpoint["online_state_dict"])
    agent.online_net.eval()
    return agent, env, checkpoint


def resolve_marker(stats: dict, name: str) -> float:
    value = stats.get(name)
    if value is None:
        return float(stats.get("rounds_survived", 0))
    return float(value)


def run_evaluation(upstream: Path, episodes: int, seed: int, max_rounds: int) -> list[dict]:
    set_global_seed(seed)
    agent, env, _ = make_upstream_components(upstream, max_rounds)
    rows = []
    for episode in range(episodes):
        trial_seed = seed + episode
        state, info = env.reset(seed=trial_seed)
        edge_index, edge_weight = env.get_graph_data()
        alive_mask = env.get_alive_mask()
        done = False
        reward_total = 0.0
        while not done:
            action, _ = agent.select_action(
                state, edge_index, edge_weight, alive_mask
            )
            state, reward, terminated, truncated, info = env.step(action)
            edge_index, edge_weight = env.get_graph_data()
            alive_mask = env.get_alive_mask()
            reward_total += (
                float(np.mean(reward))
                if isinstance(reward, np.ndarray)
                else float(reward)
            )
            done = terminated or truncated

        stats = info.get("episode_stats", {})
        total_energy = (
            float(stats.get("total_tx_cost", 0.0))
            + float(stats.get("total_recluster_cost", 0.0))
        )
        packets = float(stats.get("packets_delivered", 0.0))
        rounds = float(stats.get("rounds_survived", 0.0))
        row = {
            "trial": episode + 1,
            "seed": trial_seed,
            "t_fnd": resolve_marker(stats, "t_fnd"),
            "t_hnd": resolve_marker(stats, "t_hnd"),
            "rounds": rounds,
            "throughput": packets * 4000.0 / max(rounds, 1.0),
            "energy_efficiency": packets / max(total_energy, 1e-8),
            "reward": reward_total,
            "alive_at_end": int(info.get("alive_count", 0)),
        }
        rows.append(row)
        print(
            f"trial={row['trial']:02d} seed={trial_seed} "
            f"T_FND={row['t_fnd']:.1f} T_HND={row['t_hnd']:.1f} "
            f"rounds={row['rounds']:.1f}",
            flush=True,
        )
    return rows


def summarize(rows: list[dict], config: dict) -> tuple[dict, bool]:
    metrics = ("t_fnd", "t_hnd", "rounds", "throughput", "energy_efficiency")
    summary = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        summary[metric] = {
            "mean": float(values.mean()),
            "population_std": float(values.std(ddof=0)),
            "median": float(median),
            "iqr": float(q3 - q1),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    reference = config["evaluation"]
    observed = summary["t_fnd"]
    combined_se = np.sqrt(
        reference["reference_t_fnd_std"] ** 2 / reference["reference_trials"]
        + observed["population_std"] ** 2 / len(rows)
    )
    threshold = 1.96 * combined_se
    difference = abs(observed["mean"] - reference["reference_t_fnd_mean"])
    summary["t_fnd_reproduction_test"] = {
        "method": "absolute mean difference <= 1.96 * combined standard error",
        "absolute_difference": float(difference),
        "threshold": float(threshold),
        "pass": bool(difference <= threshold),
    }
    return summary, bool(difference <= threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=PROJECT_ROOT.parent / "final_repo")
    parser.add_argument("--run-evaluation", action="store_true")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-rounds", type=int, default=2500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    upstream = args.upstream.resolve()
    config = load_yaml(BASE_CONFIG_PATH)
    manifest = load_yaml(ASSET_MANIFEST_PATH)
    failures = validate_base_config(config)

    current_commit = git_commit(upstream)
    if current_commit != manifest["upstream"]["git_commit"]:
        failures.append(
            f"upstream commit {current_commit!r} does not match frozen manifest"
        )

    for asset_name in ("checkpoint", "solar_hmm"):
        spec = manifest[asset_name]
        path = upstream / spec["path"]
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            failures.append(f"{asset_name} SHA-256 mismatch: {actual_hash}")

    hmm_values, hmm_failures = inspect_hmm_artifacts(upstream, manifest)
    failures.extend(hmm_failures)

    evaluation = None
    evaluation_pass = None
    if args.run_evaluation:
        if args.episodes != config["evaluation"]["phase0_trials"]:
            failures.append(
                f"Phase 0 requires exactly {config['evaluation']['phase0_trials']} trials"
            )
        rows = run_evaluation(upstream, args.episodes, args.seed, args.max_rounds)
        evaluation, evaluation_pass = summarize(rows, config)
        if not evaluation_pass:
            failures.append(
                "fresh checkpoint T_FND does not reproduce the locked reference "
                "under the preregistered combined-standard-error criterion"
            )
    else:
        rows = []

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 0,
        "gate_pass": not failures and args.run_evaluation,
        "evaluation_executed": args.run_evaluation,
        "failures": failures,
        "reproducibility": {
            "upstream_path": str(upstream),
            "upstream_git_commit": current_commit,
            "base_config_sha256": sha256_file(BASE_CONFIG_PATH),
            "checkpoint_sha256": sha256_file(
                upstream / manifest["checkpoint"]["path"]
            ),
            "solar_hmm_sha256": sha256_file(
                upstream / manifest["solar_hmm"]["path"]
            ),
            "seed_start": args.seed,
            "episodes": args.episodes if args.run_evaluation else 0,
            "max_rounds": args.max_rounds,
        },
        "hmm_parameters": hmm_values,
        "evaluation_summary": evaluation,
        "trials": rows,
    }
    output_dir = PROJECT_ROOT / "outputs" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"phase0_gate_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"base_config_sha256={report['reproducibility']['base_config_sha256']}")
    print(f"upstream_git_commit={current_commit}")
    print(f"solar_A={np.asarray(hmm_values['solar'].get('hmm_A'))}")
    print(f"solar_mu={np.asarray(hmm_values['solar'].get('hmm_mu'))}")
    print(f"solar_sigma2={np.asarray(hmm_values['solar'].get('hmm_sigma2'))}")
    print(f"report={output_path}")
    print(f"PHASE_0_GATE={'PASS' if report['gate_pass'] else 'FAIL'}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
