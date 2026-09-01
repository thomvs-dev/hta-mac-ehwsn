# HTA-MAC Load-Robust Fair-Energy Metric

**Date:** 17 August 2026  
**Status:** implemented; development-pilot discovery only  
**Confirmation cohort:** environment seeds 4200--4219, unopened

## Decision

It would be methodologically invalid to invent a weighted score solely because
it makes HTA-MAC win. The implemented measure instead targets a specific and
physically interpretable property of adaptive MAC control: retaining fair,
energy-efficient service when offered traffic increases. Standard delivery,
staleness, Jain fairness, raw packets/J, latency, FND, HND, and censoring remain
mandatory. The derived measure cannot replace or conceal them.

## Literature basis

Meshkati et al. define wireless energy-efficiency utility as reliable goodput
divided by power while treating delay requirements as QoS constraints. Gao et
al. similarly optimize energy efficiency subject to statistical QoS in an
energy-harvesting WSN. These support packets/J under QoS, rather than an
uninterpretable reward score.

Yang et al. use alpha-fair allocation and Jain's index to expose the sum-rate
versus fairness trade-off in a wireless-powered network. REE-MAC separately
evaluates energy-aware MAC scheduling and fairness. These support retaining an
explicit node-service fairness term.

Jaffres-Runser et al. argue that wireless sensor/ad-hoc networks are not
adequately characterized by one metric and present Pareto sets over energy,
delay, and robustness. Related work on energy-efficient resource allocation
with QoS constraints and robust WSN design under traffic uncertainty likewise
supports stress-scenario evaluation. None of these papers defines the exact
retention ratio below; LRFER is therefore presented as a new, secondary,
project-specific metric, not as an established standard.

Key sources:

1. F. Meshkati et al., [Energy Efficiency and Delay Quality-of-Service in Wireless Networks](https://arxiv.org/abs/cs/0601098), 2006/2007.
2. K. Jaffres-Runser et al., [A Multiobjective Optimization Framework for Routing in Wireless Ad Hoc Networks](https://arxiv.org/abs/0902.0782), 2010.
3. Z. Yang et al., [Optimal Fairness-Aware Time and Power Allocation in Wireless Powered Communication Networks](https://arxiv.org/abs/1802.04951), 2018.
4. Y. Gao et al., [Statistical-QoS Guaranteed Energy Efficiency Optimization for Energy Harvesting Wireless Sensor Networks](https://doi.org/10.3390/s17091933), 2017.
5. S.-B. Lee et al., [Residual Energy Estimation-Based MAC Protocol for Wireless Powered Sensor Networks](https://doi.org/10.3390/s21227617), 2021.
6. F. Meshkati et al., [Energy-Efficient Resource Allocation in Wireless Networks with Quality-of-Service Constraints](https://arxiv.org/abs/0704.3880), 2007.
7. K. Jaffres-Runser et al., [On the Performance Evaluation of Wireless Networks with Broadcast and Interference-Limited Channels](https://arxiv.org/abs/1009.2858), 2010.
8. V. Gabale et al., [Towards the Fast and Robust Optimal Design of Wireless Body Area Networks](https://arxiv.org/abs/1504.01356), 2015.

## Definitions

Let `G_s,p` be successful network packets per joule for policy `p` in scenario
`s`, and let `J_s,p` be Jain fairness of the per-node offered-to-served ratios.
The Jain-weighted goodput efficiency is

```text
JWGE_s,p = G_s,p * J_s,p .
```

Jain's index is dimensionless, so JWGE remains in packets/J. Both components
must also be reported separately.

For environment seed `i`, load-robust fair-energy retention is

```text
LRFER_i,p = JWGE_i,p(traffic_high) / JWGE_i,p(reference_100) .
```

The estimator is the mean of the paired seed ratios, not a ratio of unrelated
cohort means. A value of 1 denotes complete retention; 0.9 denotes a 10% loss.
Bootstrap confidence intervals and paired Wilcoxon comparisons use environment
seed as the unit. No coefficient is tuned.

QoS remains a constraint. A policy is marked jointly feasible only if delivery
is at least 0.235, stale ratio is at most 0.45, and service fairness is at least
0.70 under the frozen QoS contract.

## Development-pilot result

The five already-open seeds 4100--4104 give the following exploratory ranking:

| Policy | Mean LRFER | Interpretation |
|---|---:|---|
| HTA-MAC | **0.9561** | retained 95.61% |
| HTA-MAC without CH context | 0.9541 | retained 95.41% |
| HTA-MAC without set context | 0.9449 | retained 94.49% |
| Online primal-dual | 0.9274 | retained 92.74% |
| Energy-proportional | 0.8609 | retained 86.09% |

HTA-MAC minus energy-proportional retention is 0.0952 with a pilot bootstrap
95% interval of approximately [0.06, 0.13]. HTA-MAC minus primal-dual is 0.0287
with an interval of approximately [0.01, 0.05]. Exact two-sided Wilcoxon
p-values are 0.0625 because n=5 is too small for a conventional 0.05 claim.
HTA-MAC is not distinguishable from the no-CH-context intervention in this
pilot. The result is therefore a confirmation hypothesis, not evidence of
general superiority or of CH-context necessity.

JWGE itself does not make HTA-MAC uniformly best: the energy-proportional and
primal-dual baselines remain ahead in the 20-node scenario. That negative result
must remain in the paper.

## Implementation and safeguards

- Every new trial records JWGE and frozen-QoS joint feasibility.
- Analysis computes LRFER from paired reference/high-traffic seeds.
- The contract binds confirmation to seeds 4200--4219.
- Existing pilot evidence is reanalysed without simulation or mutation and is
  linked by SHA-256.
- Raw and component metrics remain in every result.
- FND/HND and censored LND remain separate; lifetime is not hidden inside the
  new score.
- No best optimizer seed or checkpoint may be selected using LRFER.
- The failed 15-lineage retraining matrix is not rescued by this metric; model
  repair and gate passage are still required before final confirmation.

## Publication use

The defensible claim, if the sealed cohort confirms it, is narrow:

> HTA-MAC preserves a larger fraction of fair goodput-per-joule under increased
> offered load than the declared energy-proportional and online primal-dual
> baselines, while exhibiting an explicit early-FND trade-off.

Do not write that HTA-MAC is universally best, has superior lifetime, or wins
all operating regimes.

## Rerun inputs

```text
workflow: firecrawl-research-papers
topic: fair energy efficiency and robustness under traffic load in EH-WSN MAC
target_count: related metric family plus verified load-bearing definitions
output: metric design, implementation, and development-pilot analysis
```
