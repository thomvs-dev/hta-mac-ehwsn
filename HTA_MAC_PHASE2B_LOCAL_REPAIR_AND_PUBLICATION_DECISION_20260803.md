# HTA-MAC Phase 2B Local Repair and Publication Decision

**Date:** 3 August 2026  
**Scope:** MAC-only HTA-MAC under frozen HEART-CH schedule replay  
**Status:** Development repair candidate; **not a publication-ready checkpoint**

## 1. Executive decision

The registered Phase 2 sweep is technically intact, but its learned policy showed weak decision-level dependence on the claimed harvest-trajectory input. A short local repair was therefore tested instead of launching another 10–12 hour sweep.

The repair produced a real mechanistic improvement on a paired mid-episode development diagnostic:

- Complete high-versus-low hybrid-trajectory marginal ordering increased from **189/3,809 (4.96%)** to **1,806/3,809 (47.41%)**.
- The probed node's projected slot count changed in **14/3,809 (0.37%)** registered-policy cases and **50/3,809 (1.31%)** repaired-policy cases.
- Direction changed from **6 high-more versus 8 high-fewer** to **40 high-more versus 10 high-fewer**.
- Greedy development evaluation retained high packet delivery and did not collapse to always-sleep.

This is sufficient to keep the repaired formulation as the Phase 2B candidate. It is not sufficient to claim publication-ready superiority because the experiment used one optimizer lineage, development seeds, only 20+10 continuation episodes, and no paired 30-trial held-out evaluation.

## 2. Why a repair was necessary

The registered checkpoint passed the implemented Phase 2 gate, but the later policy-level audit exposed a mismatch between the gate and the scientific claim:

- The original gate treated any numerical S1/S8 Q difference above a small threshold as differentiation.
- Across the registered sweep, almost no budget-projected decisions changed under the counterfactual.
- In the registered budget-12 seed-2299 checkpoint, the reset-state solar counterfactual changed no local argmax and no projected allocation in 98 probes.
- The harvest mean and variance features were many orders of magnitude smaller than the inherited embedding values. A single global LayerNorm over the flattened shared-network input did not give those physical features comparable influence.

The correct response was to repair feature representation and test decision-level behavior, not to reinterpret a small Q difference as proof of trajectory-aware scheduling.

## 3. Precision decision

Local hardware inspection found:

- PyTorch: 2.11.0+cpu
- CUDA: unavailable
- CPU capability reported by PyTorch: AVX2
- No native AVX512-BF16 or AMX-BF16 execution path was available

A real forward/backward benchmark on the HTA-MAC model gave:

| Precision | Seconds per step |
|---|---:|
| FP32 | 0.046243 |
| CPU BF16 autocast | 0.115329 |

BF16 was **2.49 times slower** locally. Local work therefore remained FP32. BF16 should only be selected on a GPU after torch.cuda.is_bf16_supported() returns true and an actual HTA-MAC step benchmark confirms a speed benefit. Precision does not repair policy semantics.

## 4. Implemented repair

### 4.1 Physical feature scaling

For each node, the hybrid next-harvest mean and variance are transformed as

\[
\widetilde{\mu}_{H}=\frac{\mu_H}{H_{\mathrm{ref}}},
\qquad
\widetilde{\sigma^2}_{H}=\frac{\sigma^2_H}{H_{\mathrm{ref}}^2}.
\]

Here, H_ref is the largest state-conditioned combined solar-plus-thermal expected harvest produced by the frozen HMM assets.

The inherited 32-dimensional embedding block is normalized per node. Energy, queue, previous allocation, cluster-size fraction, and the state-conditioned transition-probability blocks retain their existing bounded semantics. The transform is opt-in and stored in the checkpoint configuration. Old checkpoints remain loadable because the default is disabled.

### 4.2 Hybrid trajectory-order regularizer

For an active node at equal residual energy, two counterfactual states are generated:

- low: lowest solar and thermal state-conditioned transition rows and corresponding rectified moments;
- high: highest solar and thermal state-conditioned transition rows and corresponding rectified moments.

For slot marginal value ΔQ_a = Q(a) - Q(a-1), the auxiliary loss is

\[
\mathcal{L}_{\mathrm{traj}}
=
\frac{1}{A-1}
\sum_a
\max\left(0,\eta s+\Delta Q_a^{\mathrm{low}}
-\Delta Q_a^{\mathrm{high}}\right),
\]

where s is a detached robust scale from the current marginal-Q magnitudes and η = 0.05.

This is a soft inductive bias, not a hard claim that high harvest must always receive more slots. Queue caps, energy, fairness, and the global budget can still make sleeping or a smaller allocation optimal.

### 4.3 Diminishing-return regularizer

Successive slot gains are encouraged to be non-increasing:

\[
\mathcal{L}_{\mathrm{concave}}
=
\frac{1}{A-2}
\sum_a
\max(0,\Delta Q_{a+1}-\Delta Q_a).
\]

This aligns the branch values with the marginal-gain budget projection and discourages a preference for later slots whose earlier increments have lower value.

### 4.4 Total optimization objective

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{C51}}
+\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}}
+\lambda_{\mathrm{concave}}\mathcal{L}_{\mathrm{concave}}.
\]

Every component is logged separately. The repair does not alter the frozen CH schedule, HMM parameters, radio model, queue rules, action budget, or reward definition.

## 5. Verification

The complete local validation suite passed after implementation:

    43 passed, 93 warnings in 6.73 s

The warnings were dependency deprecations; no test failed and no non-finite result occurred.

Files changed or added:

- agents/branching_dqn.py
- experiments/train_phase2_dynamic_curriculum.py
- experiments/train_phase2_fixed_cluster.py
- experiments/audit_phase2_trajectory_sensitivity.py
- experiments/audit_phase2_mid_episode_hybrid_sensitivity.py
- validation/test_phase2_agent.py

## 6. Short-run evidence

### 6.1 Run A: 20-episode warm-start screen

- Initialization: registered shared-branching, budget 12, seed 2299
- Learning rate: 2e-5
- Exploration: 0.15 to 0.05
- trajectory weight: 0.10
- concavity weight: 0.05
- Runtime: 171.9 s
- Always-sleep collapse: false
- Non-finite failure: false

### 6.2 Run B: 10-episode conservative adjustment

- Initialization: Run A checkpoint
- Learning rate: 1e-5
- Exploration: 0.10 to 0.03
- trajectory weight: 1.00
- concavity weight: 0.10
- Runtime: 107.9 s
- Always-sleep collapse: false
- Non-finite failure: false

The stronger trajectory weight reduced the logged trajectory-order loss, but it did not make reset-state probes informative because their queue-derived action caps were normally one. Weight escalation was stopped after this single adjustment.

### 6.3 Greedy development metrics

These metrics share the same development curriculum and are useful for screening only.

| Checkpoint | Mean target packets | Mean FND-free steps | Mean global throughput | Queue fairness | Delivery ratio |
|---|---:|---:|---:|---:|---:|
| Registered budget-12 seed-2299 | 1223.44 | 141.28 | 10409.48 | 0.43130 | 0.67298 |
| Run A, 20 episodes | 1464.20 | 139.88 | 10460.88 | 0.58505 | 0.68966 |
| Run B, +10 episodes | 1455.92 | 139.72 | 10456.00 | 0.57359 | 0.68930 |

Interpretation: the repaired candidate improves packet service, delivery ratio, and fairness in this screen, with a small FND-free-step decrease. This is a trade-off, not unconditional dominance.

Neither short run passed the formal Phase 2 curriculum gate. Both are correctly labelled smoke runs: convergence and multi-snapshot stability were not established, and fewer than 25 episodes cannot visit the full 25-cluster curriculum.

## 7. Corrected decision-level audit

The original reset-only audit was insufficient for a multi-slot claim because queue feasibility usually capped a freshly reset node at one packet. A new audit therefore collects mid-episode states where the probed node has a queue-derived cap of at least two.

Protocol:

- five development seeds: 2300–2304;
- 25 frozen-schedule cluster environments;
- maximum 60 rollout rounds per environment;
- up to three backlog-eligible nodes per state;
- deterministic audit seed 20260803;
- identical states, masks, caps, and counterfactuals for both checkpoints;
- equal normalized residual energy;
- lowest joint solar+thermal block versus highest joint solar+thermal block;
- 3,809 paired probes.

| Diagnostic | Registered | Repaired Run B |
|---|---:|---:|
| Local argmax changed | 16 (0.42%) | 64 (1.68%) |
| Probed-node projected slots changed | 14 (0.37%) | 50 (1.31%) |
| Joint projected vector changed | 105 (2.76%) | 448 (11.76%) |
| High trajectory received more slots | 6 (0.16%) | 40 (1.05%) |
| High trajectory received fewer slots | 8 (0.21%) | 10 (0.26%) |
| All marginal gains correctly ordered | 189 (4.96%) | 1806 (47.41%) |

Archived evidence:

- outputs/phase2/phase2b_mid_episode_hybrid_sensitivity.json
- outputs/phase2/phase2b_local_screen_b12_seed2299_20ep/
- outputs/phase2/phase2b_local_adjusted_b12_seed2299_10ep/

## 8. What the evidence does and does not establish

It establishes that the repaired network is substantially more sensitive in the intended direction on backlog-eligible hybrid-trajectory counterfactuals, without an immediate always-sleep collapse.

It does not establish:

- convergence;
- optimizer-seed robustness;
- held-out generalization;
- statistical superiority over baselines;
- lifetime improvement;
- publication-ready HTA-MAC performance;
- strict monotonic behavior in every state;
- C4 lifetime-bound dominance.

The absolute projected-node change rate remains low at 1.31%. This may be legitimate because many states are not close to a budget decision boundary, but it must be studied rather than hidden.

## 9. Research basis and novelty discipline

The implementation keeps the defensible novelty horizon identified in the literature audits: a shared Branching Dueling value network for budget-constrained discrete per-node multi-slot allocation using joint solar/thermal HMM trajectory features under frozen CH schedules.

Relevant anchors include:

- Tavakoli, Pardo, and Kormushev, Branching Dueling Q-Networks: https://arxiv.org/abs/1711.08946
- SHR-TDMA hybrid-EH allocation: https://doi.org/10.1049/iet-com.2019.0977
- FFSS/AFSS forecast-aware scheduling: https://doi.org/10.1049/cmu2.12243
- Q-learning TDMA scheduling: https://doi.org/10.1016/j.comcom.2022.08.013
- Contextual DRL joint slot/sleep scheduling: https://doi.org/10.1109/TGCN.2024.3358230
- S2A2MAC HMM-guided MAC: https://doi.org/10.1002/dac.5202
- Structure-aware EH control literature: https://doi.org/10.1109/TSP.2020.2973125
- Constrained monotonic neural-network literature: https://proceedings.mlr.press/v202/runje23a.html

These references motivate architecture and inductive bias. They do not prove HTA-MAC performance.

## 10. Minimum publication path from here

Do not repeat the 18-run architecture/budget sweep yet. The time-efficient next experiment is:

1. Freeze this Phase 2B formulation and its diagnostic before further training.
2. Train only the chosen shared-branching budget-12 model for three optimizer seeds.
3. Use at least 25 episodes so every curriculum pair is visited; 100–150 episodes per seed is a practical first convergence screen.
4. Monitor predeclared quantities:
   - no non-finite or always-sleep collapse;
   - full curriculum coverage;
   - C51, trajectory, and concavity losses separately;
   - development performance trade-off;
   - paired backlog-eligible hybrid sensitivity on a fixed probe bank;
   - stability snapshots.
5. If seed behavior is consistent, evaluate the frozen three checkpoints on untouched held-out schedules and then run the declared paired 30-trial baseline comparison.
6. If per-node sensitivity remains rare or unstable, report the registered system as a negative diagnostic result and revise the architecture before manuscript claims. Do not proceed to Phase 4 with a claim unsupported at the decision layer.

A practical stop rule should be pre-registered before the next run. At minimum, the repaired model must show a consistent high-more versus high-fewer directional imbalance across all three optimizer seeds without more than the predeclared acceptable degradation in FND-free steps. The tolerance must be fixed before viewing those results.

## 11. Remaining doubts

- Whether 1.31% projected-node changes are enough to create statistically detectable lifetime differences.
- Whether the repair generalizes beyond the single seed-2299 optimizer lineage.
- Whether branch-index-specific heads learn node identity artifacts under frozen global node indexing.
- Whether the soft monotonic prior is too broad: physically, high future harvest may justify either more immediate service or strategic deferral depending on queue urgency and horizon.
- Whether the FND loss persists under longer training or is only a short-run perturbation.
- Whether a next-harvest auxiliary prediction head would improve representation more cleanly than further increasing the trajectory-order weight.
- Whether the thermal default model is strong enough for more than a mechanism-level hybrid-source claim.

## 12. Final recommendation

Keep the repaired checkpoint as **Phase 2B candidate R1**, not as the paper model. The short local experiment succeeded at its intended purpose: it found a measurable, directionally correct policy-level response and exposed why the reset-only audit understated multi-slot behavior. The next authorized compute should be a narrow three-seed budget-12 confirmation, not another broad sweep and not Phase 4.