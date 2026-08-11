"""Phase 3 heuristic, literature-adapted, learned, and diagnostic policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.budget_projection import project_slot_budget
from baselines.interface import MACPolicyInterface
from envs.policy_observation import (
    PHASE2D_POLICY_SCHEMA,
    build_policy_observation,
)


def _rank_proportional(scores, budget: int, n_max: int) -> np.ndarray:
    """Discrete capped proportional allocation through successive quotients."""
    scores = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    allocation = np.zeros(len(scores), dtype=np.int64)
    if not len(scores) or budget <= 0 or not np.any(scores > 0.0):
        return allocation
    for _ in range(int(budget)):
        eligible = allocation < int(n_max)
        if not np.any(eligible):
            break
        quotient = np.full(len(scores), -np.inf, dtype=np.float64)
        quotient[eligible] = scores[eligible] / (allocation[eligible] + 1.0)
        selected = int(np.argmax(quotient))
        if not np.isfinite(quotient[selected]) or quotient[selected] <= 0.0:
            break
        allocation[selected] += 1
    return allocation


class StaticEqualPolicy(MACPolicyInterface):
    name = "static_equal"

    def select_action(self, state, env):
        return self.validate(env.static_equal_action(), env)


class EnergyProportionalPolicy(MACPolicyInterface):
    name = "energy_proportional"

    def __init__(self, score_exponent: float = 1.0):
        if score_exponent <= 0.0:
            raise ValueError("energy score exponent must be positive")
        self.score_exponent = float(score_exponent)

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        for cluster, ch in enumerate(env.cluster_heads):
            members = self.eligible_members(env, cluster, int(ch))
            action[members] = _rank_proportional(
                np.power(
                    env.energy[members] / env.cfg.initial_energy_j,
                    self.score_exponent,
                ),
                env.cfg.frame_slot_budget,
                env.cfg.n_max,
            )
        return self.validate(action, env)


class HarvestProportionalPolicy(MACPolicyInterface):
    name = "harvest_proportional"

    def __init__(self, score_exponent: float = 1.0):
        if score_exponent <= 0.0:
            raise ValueError("harvest score exponent must be positive")
        self.score_exponent = float(score_exponent)

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        forecast = np.maximum(np.asarray(state[:, 1], dtype=np.float64), 0.0)
        for cluster, ch in enumerate(env.cluster_heads):
            members = self.eligible_members(env, cluster, int(ch))
            scores = np.power(forecast[members], self.score_exponent)
            if len(scores) and not np.any(scores > 0.0):
                scores = np.ones(len(members), dtype=np.float64)
            action[members] = _rank_proportional(
                scores, env.cfg.frame_slot_budget, env.cfg.n_max
            )
        return self.validate(action, env)


class S2A2MACAdaptedPolicy(MACPolicyInterface):
    """Documented structural adaptation of Movva et al.'s S2A2MAC.

    Odd/even clusters alternate sleep and active rounds. Within active clusters,
    residual energy and queue load form three node layers receiving 1/2/3
    mini-slots. The source does not publish reusable HMM parameters, so the
    three categories are reproduced by deterministic cluster-local tertiles.
    """

    name = "s2a2mac_adapted"

    def __init__(self, energy_weight: float = 0.5):
        if not 0.0 <= energy_weight <= 1.0:
            raise ValueError("S2A2MAC energy weight must be in [0, 1]")
        self.energy_weight = float(energy_weight)

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        for cluster, ch in enumerate(env.cluster_heads):
            if (cluster + env.round) % 2 == 0:
                continue
            members = self.eligible_members(env, cluster, int(ch))
            if not len(members):
                continue
            energy = np.clip(
                env.energy[members] / env.cfg.initial_energy_j, 0.0, 1.0
            )
            load = np.clip(
                env.queue[members] / env.cfg.queue_max_packets, 0.0, 1.0
            )
            score = (
                self.energy_weight * energy
                + (1.0 - self.energy_weight) * load
            )
            order = np.argsort(score, kind="stable")
            layers = np.ones(len(members), dtype=np.int64)
            first = len(members) // 3
            second = (2 * len(members)) // 3
            layers[order[first:second]] = 2
            layers[order[second:]] = 3
            q_values = np.zeros((len(members), env.cfg.n_max + 1))
            for local, desired in enumerate(layers):
                q_values[local, : desired + 1] = np.arange(desired + 1)
                if desired < env.cfg.n_max:
                    q_values[local, desired + 1 :] = desired - 1.0
            action[members] = project_slot_budget(
                q_values,
                env.cfg.frame_slot_budget,
                stop_at_nonpositive_gain=True,
            )
        return self.validate(action, env)


class FFSSAdaptedPolicy(MACPolicyInterface):
    """Fixed-frame feasible-first adaptation of Gong et al.'s FFSS.

    FFSS assigns one slot per node and optimizes within-frame order using future
    energy/data. The current round-level environment has no slot-order state, so
    this adaptation retains one-slot assignment and feasible-first selection
    when cluster membership exceeds the fixed frame budget.
    """

    name = "ffss_adapted"

    def __init__(
        self,
        margin_weight: float = 1.0,
        queue_weight: float = 1.0,
    ):
        if margin_weight < 0.0 or queue_weight < 0.0:
            raise ValueError("FFSS adaptation weights must be nonnegative")
        if margin_weight + queue_weight == 0.0:
            raise ValueError("at least one FFSS adaptation weight must be positive")
        self.margin_weight = float(margin_weight)
        self.queue_weight = float(queue_weight)

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        forecast = np.maximum(np.asarray(state[:, 1], dtype=np.float64), 0.0)
        for cluster, ch in enumerate(env.cluster_heads):
            members = self.eligible_members(env, cluster, int(ch))
            if not len(members):
                continue
            distance = np.linalg.norm(
                env.positions[members] - env.positions[int(ch)], axis=1
            )
            required = np.asarray(
                [env.radio.tx(env.cfg.packet_bits, float(d)) for d in distance]
            )
            available = env.energy[members] + forecast[members]
            has_data = env.queue[members] > 0
            margin = available - required
            qualified = has_data & (margin >= 0.0)
            margin_scale = max(float(np.max(np.abs(margin))), 1e-12)
            normalized_margin = margin / margin_scale
            normalized_queue = (
                env.queue[members] / env.cfg.queue_max_packets
            )
            score = (
                self.margin_weight * normalized_margin
                + self.queue_weight * normalized_queue
            )
            priority = np.lexsort((members, -score, ~qualified))
            selected = members[priority[: env.cfg.frame_slot_budget]]
            action[selected] = 1
        return self.validate(action, env)


class RandomBudgetedPolicy(MACPolicyInterface):
    name = "random_budgeted_diagnostic"
    literature_baseline = False
    comparison_role = "formal_stochastic_floor"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        for cluster, ch in enumerate(env.cluster_heads):
            members = self.eligible_members(env, cluster, int(ch))
            q_values = self.rng.normal(
                size=(len(members), env.cfg.n_max + 1)
            )
            if len(members):
                q_values[:, 1:] += 0.5
                action[members] = project_slot_budget(
                    q_values,
                    env.cfg.frame_slot_budget,
                    stop_at_nonpositive_gain=False,
                )
        return self.validate(action, env)


class HTAMACPolicy(MACPolicyInterface):
    name = "hta_mac"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        allocation_budget: int | None = None,
    ):
        self.device = device
        self.allocation_budget = allocation_budget
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=False
        )
        raw_config = dict(checkpoint["config"])
        if "architecture" not in raw_config:
            raw_config["architecture"] = "legacy_weight_tied"
        config = BranchingAgentConfig(**raw_config)
        self.agent = BranchingDQNAgent(config, device=device)
        self.agent.online.load_state_dict(checkpoint["online_state_dict"])
        self.agent.target.load_state_dict(checkpoint["target_state_dict"])
        self.agent.online.eval()

    def select_action(self, state, env):
        action = np.zeros(env.n_nodes, dtype=np.int64)
        embedding = np.asarray(env.embedding, dtype=np.float32)
        legacy_features = np.concatenate((state, embedding), axis=1).astype(np.float32)
        for cluster, ch in enumerate(env.cluster_heads):
            members = self.eligible_members(env, cluster, int(ch))
            if not len(members):
                continue
            budget = min(
                env.cfg.frame_slot_budget,
                self.agent.cfg.budget
                if self.allocation_budget is None
                else self.allocation_budget,
            )
            if self.agent.cfg.architecture == "legacy_weight_tied":
                cluster_action, _ = self.agent.act(
                    legacy_features[members],
                    np.ones(len(members), dtype=bool),
                    epsilon=0.0,
                    caps=np.minimum(env.queue[members], env.cfg.n_max),
                    budget=budget,
                )
                action[members] = cluster_action
            else:
                cluster_mask = np.zeros(env.n_nodes, dtype=bool)
                cluster_mask[members] = True
                if self.agent.cfg.state_schema == PHASE2D_POLICY_SCHEMA:
                    features = build_policy_observation(
                        env,
                        state,
                        cluster_mask,
                        schema=self.agent.cfg.state_schema,
                    )
                else:
                    features = legacy_features
                caps = np.minimum(env.queue, env.cfg.n_max)
                caps[~cluster_mask] = 0
                global_action, _ = self.agent.act(
                    features,
                    cluster_mask,
                    epsilon=0.0,
                    caps=caps,
                    budget=budget,
                    tie_break_priorities=np.arange(
                        env.n_nodes, dtype=np.int64
                    ),
                )
                action[members] = global_action[members]
        return self.validate(action, env)


def phase3_policy_factories(
    root: Path,
    *,
    hta_checkpoint: str | Path | None = None,
    hta_budget: int | None = None,
):
    checkpoint = (
        root
        / "outputs"
        / "phase2"
        / "authoritative_dynamic_budget8_500ep"
        / "branching_c51.pt"
        if hta_checkpoint is None
        else Path(hta_checkpoint)
    )
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    allocation_budget = 8 if hta_budget is None else int(hta_budget)
    if allocation_budget <= 0:
        raise ValueError("HTA allocation budget must be positive")
    return (
        (StaticEqualPolicy.name, StaticEqualPolicy),
        (
            EnergyProportionalPolicy.name,
            lambda: EnergyProportionalPolicy(score_exponent=2.0),
        ),
        (
            HarvestProportionalPolicy.name,
            lambda: HarvestProportionalPolicy(score_exponent=2.0),
        ),
        (
            S2A2MACAdaptedPolicy.name,
            lambda: S2A2MACAdaptedPolicy(energy_weight=0.25),
        ),
        (
            FFSSAdaptedPolicy.name,
            lambda: FFSSAdaptedPolicy(
                margin_weight=1.0, queue_weight=0.0
            ),
        ),
        (
            HTAMACPolicy.name,
            lambda: HTAMACPolicy(
                checkpoint, allocation_budget=allocation_budget
            ),
        ),
        (RandomBudgetedPolicy.name, RandomBudgetedPolicy),
    )
