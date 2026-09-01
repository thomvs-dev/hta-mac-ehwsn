# Paper-writing claims for HTA-MAC V1

## Claims supported by this release

- HTA-MAC performs learned intra-cluster MAC allocation under an exogenous
  cluster-head schedule; it does not learn clustering or routing.
- Its shared set-based branching policy is designed for permutation-equivariant
  allocation over variable member sets.
- Under the frozen 100-node transfer protocol, HTA-MAC has lower stale loss and
  higher service fairness than the cap-corrected residual-energy heuristic.
- Under that same protocol, the corrected heuristic has higher delivery,
  restricted-mean survival, and packets/J than HTA-MAC.
- The custom online primal-dual controller is a locally implemented non-neural
  comparator. It is not a reproduction of PPO-Lagrangian, CPO, or any named
  external paper.
- PVGIS supplies a real irradiance trace, but the network and radio behavior
  remain simulated.
- Confidence intervals and paired tests apply only to the declared paired
  simulator units and metrics in the saved result artifacts.

## Historical result versus current conclusion

The original 20-seed confirmation used the preregistered comparator code and
must remain available as historical evidence. A later audit found that the
energy-proportional comparator needed cap correction. The corrected result
supersedes the original energy-proportional ranking. It does not alter the
saved HTA-MAC rollouts or silently redefine the original experiment.

For the manuscript, use the corrected 100-node values:

| Policy | Delivery | Stale loss | Fairness | RMST | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC V1 | 0.42770 | 0.02476 | 0.94511 | 128.28 | 225.77 |
| Corrected energy heuristic | 0.44591 | 0.04838 | 0.87166 | 149.32 | 242.36 |
| Custom online primal-dual | 0.43020 | 0.01700 | 0.98081 | 131.10 | 232.45 |

## Claims this release does not support

- Do not call HTA-MAC universally superior, state of the art, or best on all
  metrics.
- Do not compare numerical percentages from incompatible papers as a common
  leaderboard.
- Do not claim measured hardware inference energy or deployment performance.
- Do not call the PVGIS experiment a real-network experiment.
- Do not describe the custom primal-dual comparator as a published algorithm
  reproduction unless an implementation-faithful reproduction is added later.
- Do not claim that the checkpoint-producing curriculum gate passed.
- Do not use confirmation seeds 3900--3919 for future development.

## Recommended concise result wording

"In the matched transfer simulator, HTA-MAC shifts the operating point toward
fairer and less stale service. Against the cap-corrected residual-energy
heuristic, this benefit is accompanied by lower delivery, lifetime, and energy
efficiency, establishing a measurable trade-off rather than universal
dominance."

