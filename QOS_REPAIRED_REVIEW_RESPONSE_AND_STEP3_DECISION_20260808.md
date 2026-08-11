# QoS-Repaired Review Response and Step 3 Decision

**Date:** 8 August 2026

## Accepted framing correction

The central negative result is not merely “no improvement.” It is a diagnosed regression mechanism:

`300-round objective/horizon mismatch -> zero death-related reward signal -> no learned pre-death conservation behavior -> earlier FND when the policy enters the unseen lifetime regime.`

That mechanism is a defensible diagnostic contribution. It should be written as such rather than hidden in future work. The completed experiment demonstrates how a numerically stable, high-service policy can still be misaligned with a lifetime claim when the claim-bearing event is absent from training.

The metric-repair statement also needs the stronger instrument interpretation. The pre-repair controller could report 2,806 deliveries against 1,192 offered packets and clip the impossible ratio to 1.0. The repaired controller matching the old external outcome means the trustworthy instrument confirmed that the unreliable instrument happened to read correctly at this operating point. That fact was unknowable before enforcing the accounting invariant; it is not a rationalization of a null result.

## Requested FND-node check

Completed. All ten FND replays match the archived CSV exactly.

HTA-MAC and energy-proportional lose the same physical node in 4/5 paired seeds. The weak node changes with the environment seed. HTA-MAC reaches FND 10–48 rounds earlier in all five pairs, with a median paired difference of -39 rounds.

This makes the diagnosis more precise:

- the frozen schedule/topology/harvest realization usually defines the identity of the weakest node;
- the MAC policy affects when that shared node exhausts;
- CH frequency is balanced near 20%, so simple CH over-selection is not the cause;
- HTA-MAC's first death occurs while its dying node is CH in all five trials, suggesting a role-conditioned service/forwarding energy mechanism;
- a MAC-only fix can delay the failure through risk-aware service, but cannot remove upstream schedule exposure without changing research scope.

Detailed report: `FND_NODE_IDENTITY_DIAGNOSTIC_20260808.md`.

## Standing pre-training gate

Implemented in `validation/pretraining_claim_preflight.py` with a nonzero exit on failure. It checks:

1. activation of the reward term relied upon by the claim;
2. every-record delivery/demand accounting and repaired metric contract;
3. a fresh current-code permutation/foundation audit of the supplied checkpoint;
4. observation of the headline event and coverage of its required horizon.

Running it retroactively on the selected B16 training evidence blocks the lifetime claim exactly as intended: accounting passes, while death activation, FND observation, and horizon coverage fail.

The fresh Windows CPU foundation audit also exceeds the frozen Q tolerance slightly (`1.3709068e-6` versus `1e-6`) while log probability, action projection, and local argmax checks pass. The same source/checkpoint hashes passed in Colab. The threshold was not relaxed post hoc; Step 3 preflight must run on its target platform, with any cross-platform tolerance frozen in advance.

Instructions: `PRETRAINING_CLAIM_PREFLIGHT.md`.

## Step 3 decision

Proceed, but revise its objective from generic lifetime optimization to:

> Delay the seed-specific aggressive-policy weak-node failure while satisfying a frozen global delivery/stale-service floor.

Required design elements:

- a horizon/curriculum that reaches the pre-death regime;
- scheduled-CH-conditioned depletion-risk features or loss;
- separate energy accounting for member TX and CH receive/aggregate/BS forwarding;
- preserved trajectory-order and concavity losses;
- frozen service constraints and no post-hoc trade-off selection;
- repeated FND-node identity audit after training;
- no CH-policy modification inside the MAC-only experiment.

Do not spend full GPU-hours until the new standing preflight passes.
