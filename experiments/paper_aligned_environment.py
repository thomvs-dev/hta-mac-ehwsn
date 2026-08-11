"""Exploratory terrestrial-solar environment for cross-paper comparison.

This profile preserves HTA-MAC as the only learned intervention. Cluster-head
rotation is exogenous and deterministic, so this module does not train or
evaluate a CH policy. It is paper-aligned, not a reproduction of third-party
code or unpublished traces.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.hmm.solar_hmm import HMMParameters


PROFILE_SCHEMA_VERSION = 1
SCHEDULE_SCHEMA_VERSION = "paper_aligned_exogenous_leach_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_profile(path: str | Path) -> tuple[dict, dict]:
    profile_path = Path(path).resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported paper-aligned profile schema")
    if payload.get("status") != "exploratory_paper_aligned_development":
        raise ValueError("environment profile is not exploratory development")
    if payload.get("learned_intervention") != "hta_mac_only":
        raise ValueError("paper-aligned profile must keep HTA-MAC as the only learner")
    if payload.get("cluster_schedule") != "exogenous_balanced_leach_rotation":
        raise ValueError("unsupported paper-aligned cluster schedule")
    if payload.get("held_out_seeds_used") is not False:
        raise ValueError("paper-aligned profile must not use held-out seeds")

    network = payload["network"]
    harvesting = payload["harvesting"]
    mac = payload["mac"]
    required_positive = {
        "num_nodes": network["num_nodes"],
        "field_size_m": network["field_size_m"],
        "initial_energy_j": network["initial_energy_j"],
        "ch_ratio": network["ch_ratio"],
        "packet_bits": network["packet_bits"],
        "frame_slot_budget": mac["frame_slot_budget"],
        "n_max": mac["n_max"],
    }
    if any(not np.isfinite(float(value)) or float(value) <= 0 for value in required_positive.values()):
        raise ValueError("paper-aligned profile contains non-positive parameters")
    if not 0.0 < float(network["ch_ratio"]) < 1.0:
        raise ValueError("ch_ratio must lie in (0, 1)")
    if harvesting.get("solar") != "trained_stage1_hmm":
        raise ValueError("only the frozen trained solar HMM is supported")
    if harvesting.get("thermal") != "disabled_zero_energy":
        raise ValueError("paper-aligned profile must disable synthetic thermal energy")
    if payload.get("mobility") != "disabled_static_positions":
        raise ValueError("paper-aligned schedule currently supports static positions only")

    evidence = {
        "path": str(profile_path),
        "sha256": sha256_file(profile_path),
        "payload": payload,
    }
    return payload, evidence


def disabled_thermal_hmm(source: HMMParameters) -> HMMParameters:
    states = int(source.transition.shape[0])
    initial = np.zeros(states, dtype=np.float64)
    initial[0] = 1.0
    result = HMMParameters(
        transition=np.eye(states, dtype=np.float64),
        mean=np.zeros(states, dtype=np.float64),
        variance=np.zeros(states, dtype=np.float64),
        initial=initial,
        source=source.source,
        provenance="disabled_zero_energy_paper_aligned_profile",
    )
    result.validate(expected_states=states)
    return result


def configure_mac(profile: dict, config):
    network = profile["network"]
    mac = profile["mac"]
    return replace(
        config,
        initial_energy_j=float(network["initial_energy_j"]),
        packet_bits=int(network["packet_bits"]),
        control_packet_bits=int(network["control_packet_bits"]),
        frame_slot_budget=int(mac["frame_slot_budget"]),
        n_max=int(mac["n_max"]),
        queue_max_packets=int(mac["queue_max_packets"]),
        packet_ttl_rounds=int(mac["packet_ttl_rounds"]),
        solar_scale=float(profile["harvesting"]["solar_rectification_scale"]),
        thermal_scale=0.0,
        bs_position_m=tuple(float(v) for v in network["bs_position_m"]),
        idle_slot_bit_times=int(mac["idle_slot_bit_times"]),
    )


def _spatial_embedding(positions: np.ndarray, field_size: float, bs_position) -> np.ndarray:
    normalized = positions / float(field_size)
    bs = np.asarray(bs_position, dtype=np.float64)
    distance = np.linalg.norm(positions - bs[None, :], axis=1, keepdims=True)
    distance /= max(np.sqrt(2.0) * float(field_size), 1e-12)
    blocks = [normalized, distance, np.ones((len(positions), 1), dtype=np.float64)]
    for harmonic in range(1, 8):
        angle_x = 2.0 * np.pi * harmonic * normalized[:, 0:1]
        angle_y = 2.0 * np.pi * harmonic * normalized[:, 1:2]
        blocks.extend((np.sin(angle_x), np.cos(angle_x), np.sin(angle_y), np.cos(angle_y)))
    embedding = np.concatenate(blocks, axis=1).astype(np.float32)
    if embedding.shape != (len(positions), 32):
        raise RuntimeError("invalid analytic spatial embedding shape")
    return embedding


def _advance_states(rng: np.random.Generator, states: np.ndarray, transition: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(transition[states], axis=1)
    uniforms = rng.random(len(states))
    return np.sum(uniforms[:, None] > cdf, axis=1).astype(np.int64)


def paper_aligned_schedule(profile: dict, solar: HMMParameters, seed: int, horizon: int) -> dict:
    network = profile["network"]
    n_nodes = int(network["num_nodes"])
    field_size = float(network["field_size_m"])
    ch_count = max(1, int(round(n_nodes * float(network["ch_ratio"]))))
    if n_nodes % ch_count != 0:
        raise ValueError("balanced CH rotation requires num_nodes divisible by ch_count")

    rng = np.random.default_rng(int(seed))
    positions = rng.uniform(0.0, field_size, size=(n_nodes, 2))
    embedding = _spatial_embedding(positions, field_size, network["bs_position_m"])
    solar_states = rng.choice(len(solar.initial), size=n_nodes, p=solar.initial).astype(np.int64)
    thermal_states = np.zeros(n_nodes, dtype=np.int64)
    frames = []
    epoch_length = n_nodes // ch_count
    permutation = None
    for round_index in range(int(horizon)):
        if round_index % epoch_length == 0:
            permutation = rng.permutation(n_nodes)
        offset = (round_index % epoch_length) * ch_count
        cluster_heads = np.sort(permutation[offset : offset + ch_count]).astype(np.int64)
        frames.append(
            {
                "positions": positions.copy(),
                "solar_states": solar_states.copy(),
                "thermal_states": thermal_states.copy(),
                "cluster_heads": cluster_heads,
                "stgcn_embedding": embedding.copy(),
            }
        )
        solar_states = _advance_states(rng, solar_states, solar.transition)
    return {
        "frames": frames,
        "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
        "requested_horizon": int(horizon),
        "coverage_rounds": len(frames),
        "complete": len(frames) == int(horizon),
        "stop_reason": "horizon_reached",
        "generator": "exogenous_balanced_leach_rotation",
        "num_nodes": n_nodes,
        "cluster_heads_per_round": ch_count,
        "mobility": False,
        "embedding": "analytic_spatial_embedding_v1",
    }


def schedule_bundle(profile: dict, solar: HMMParameters, seed: int, horizon: int):
    result = paper_aligned_schedule(profile, solar, seed, horizon)
    bundle = dict(result["frames"][0])
    bundle["schedule"] = result["frames"]
    bundle["schedule_metadata"] = {key: value for key, value in result.items() if key != "frames"}
    return bundle, dict(bundle["schedule_metadata"])
