# Pre-Training Claim Preflight

Run `validation/pretraining_claim_preflight.py` before committing GPU-hours to any claim-bearing training sweep.

The input `episodes.jsonl` must come from a cheap, representative probe using the intended environment, reward wiring, and a horizon that can expose the headline event. The supplied checkpoint may be an initialized or short-smoke checkpoint produced by the current training code.

## Mandatory gates

1. **Reward activation:** the reward term relied upon by the claim must occur at nonzero magnitude in at least the frozen minimum number of probe records.
2. **Accounting invariant:** every probe episode must contain the repaired metric contract and satisfy `delivered <= target_packets_offered`.
3. **Current-code equivariance:** the script reruns `audit_phase2d_foundation.py` against the supplied checkpoint; an archived audit is not trusted.
4. **Endpoint coverage:** the probe must observe the headline event and its horizon must reach the frozen minimum event horizon.

Any failed gate exits nonzero and means **do not start the full training sweep**.

## Lifetime example

```powershell
python -B validation/pretraining_claim_preflight.py `
  --episodes-jsonl outputs/preflight/step3_probe/episodes.jsonl `
  --checkpoint outputs/preflight/step3_probe/branching_c51.pt `
  --environment-profile config/step3_lifetime_profile.json `
  --output outputs/preflight/step3_claim_preflight.json `
  --development-seeds 2400,2401,2402,2403,2404 `
  --required-reward-term deaths `
  --headline-event-field t_fnd `
  --minimum-nonzero-term-records 1 `
  --minimum-event-records 1 `
  --minimum-training-horizon 1080
```

The `1080` value above documents the current B16 development median and is not automatically valid for a new profile. Freeze the new profile's required horizon from independent pre-training calibration before using it.

## Current B16 self-check

Running the checklist against the completed seed-7399 training evidence correctly produces `pretraining_claim_preflight_fail`:

- accounting invariant: pass, 500/500;
- death reward activated: fail, 0 records;
- FND observed in training: fail, 0 records;
- horizon covers 1,080 rounds: fail, observed maximum 300;
- local current-code foundation gate: fail at max Q error `1.3709068e-6` versus the frozen `1e-6` threshold, while action and argmax agreement remain 1.0.

The last item is reproducible on the Windows PyTorch 2.11 CPU/MKL backend, while the archived Colab audit passed with the same source and checkpoint hashes. Do not change the threshold post hoc. Run the preflight on the actual target training platform and preregister a cross-platform absolute/relative numerical tolerance if portability is required.

Current evidence: `outputs/diagnostics/current_b16_lifetime_claim_preflight_20260808.json`.
