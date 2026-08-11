# Step 3 CH-role lifetime experiment — execution contract

Status: implemented and locally validated; GPU measurements pending.

## Locked scientific scope

Step 3 changes only HTA-MAC slot allocation. The frozen exogenous HEART-CH/LEACH schedule is neither learned nor edited. The added signal is conditionally conservative only when the currently scheduled cluster head is depleted and the proposed member service would create forwarding load. Existing trajectory-order and concavity losses remain enabled.

The mechanism is a development candidate, not evidence of lifetime superiority. B16 remains the secondary idle-listening-disabled side study. Any death activation observed here is profile-specific and must not be generalized to the primary idle-on track.

## Platform decision (closed before GPU training)

The preflight probe writes a complete runtime fingerprint: Python, OS, Torch, CUDA, cuDNN, GPU names/capabilities, and relevant package versions. Every full-training process recomputes that fingerprint and refuses to run unless it exactly matches the preflight contract. The probe and sweep therefore run in the same uninterrupted Colab runtime after dependency installation.

The Windows CPU/MKL `1.37e-6` versus Colab GPU `1e-6` Q-error discrepancy is not resolved by relaxing the gate. Cross-platform portability remains unvalidated. Restarting the runtime, changing packages, or changing GPU after preflight invalidates the contract and requires a new probe.

## New implementation

- `core/runtime_fingerprint.py`: exact runtime contract and mismatch diagnostics.
- `agents/ch_depletion_risk.py`: observable scheduled-CH risk term; no realized future-harvest leakage.
- `envs/step3_lifetime_env.py`: exact member-TX, CH-RX, CH-aggregation, CH-to-BS, and idle energy decomposition with reconstruction assertion.
- `experiments/calibrate_step3_return_scale.py`: development-only C51 return scaling including the new term.
- `experiments/train_step3_probe.py`: hard-limited 1–5 episode probe.
- `validation/step3_pretraining_preflight.py`: death, FND, QoS-accounting, horizon, risk-activation, risk-dominance, foundation, and runtime gates.
- `experiments/train_step3_lifetime.py`: full-training entry point that requires the passed preflight and exact runtime contract.

## Stop conditions

Do not start the three-seed GPU sweep if any of these is false:

1. Complete validation suite passes.
2. Current-code permutation foundation passes at the frozen `1e-6` tolerance on the actual training runtime.
3. Probe observes at least one death and at least one `t_fnd`.
4. CH-depletion risk activates at least once.
5. The risk contribution does not exceed 20% of total absolute weighted physical reward in any probe episode.
6. Every QoS row retains delivered <= demand and the repaired metric contract.
7. Probe and full-run runtime fingerprints match exactly.

If a gate fails, record the failure and redesign or recalibrate before training. Do not select the least-bad candidate and do not weaken a threshold post hoc.

## Pending measurements

No Step 3 performance result is claimed yet. The next evidence-producing action is a Colab probe followed, only on pass, by fresh optimizer seeds and paired development evaluation. Reserved confirmation seeds 3400–3404 and prohibited registered held-out seeds 3100–3104 remain untouched.
