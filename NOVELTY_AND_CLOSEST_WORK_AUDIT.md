# HTA-MAC Novelty and Closest-Work Audit

**Audit date:** 2026-07-30  
**Purpose:** Replace broad or contradicted novelty language with claims supported by inspected primary sources. This is a targeted audit, not a systematic-review proof that no paper exists.

## Executive decision

The original claim that S2A2MAC applies one active/inactive rule to an entire cluster is not supported by its primary abstract, which states that its HMM controls the active period of a **node** adaptively. The supplied verbatim differentiation sentence must not be used.

The claim that HTA-MAC is the first hybrid-harvest per-node TDMA slot allocator is also too broad. Gong et al.'s SHR-TDMA (2020) explicitly studies hybrid-source EH-WSNs, derives Markov-model-based slot-hitting ratios, and optimally assigns slots for every node. Eriş et al. (2024) independently learn per-node intra-cluster TDMA transmission-slot choices with cooperative independent Q-learning in an EH underwater network.

A defensible, test-contingent positioning is:

> Among the audited clustered EH-WSN MAC methods, HTA-MAC is distinguished by a centralized Branching Dueling Q formulation that selects a discrete per-node slot count under a cluster budget from node-specific solar-and-thermal HMM trajectory features, while keeping the upstream CH schedule fixed for paired causal attribution.

This is a differentiation statement, not an absolute worldwide first claim. It becomes a contribution only if the preregistered experiments show a useful lifetime/QoS tradeoff and the architecture ablation supports the shared-branch design.

## Primary-source comparison matrix

| Work | What the primary source supports | Overlap with HTA-MAC | Defensible distinction / required caution |
|---|---|---|---|
| Movva et al., S2A2MAC, IJCS 2022, DOI `10.1002/dac.5202` | Clustered EH-WSN; HMM adaptively controls a node's active period; also includes CH selection and routing. | HMM-aware adaptive MAC in a clustered EH-WSN; node-level behavior. | Do not say it uses only one cluster-wide rule. Audited material does not show learned Branching DQN slot-count allocation or dual solar+thermal trajectory features. Full mechanism claims require the full paper, not abstract inference. |
| Gong et al., SHR-TDMA, IET Communications 2020, DOI `10.1049/iet-com.2019.0977` | Hybrid FIAT+RIAT energy sources; Markov/number-theoretic slot-hitting ratios; delay-minimizing optimal slot allocation for every node via Hungarian assignment. | Hybrid harvest, Markov characterization, per-node TDMA assignment. | This is a close conceptual predecessor. HTA-MAC must distinguish HMM trajectory-conditioned learned slot **counts**, clustering/frozen CH context, budget projection, queues/fairness, and empirical objective—not claim hybrid per-node TDMA novelty. |
| Gong et al., FFSS/AFSS, IET Communications 2021, DOI `10.1049/cmu2.12243` | Uses upcoming energy and data; optimizes fixed/adaptive frame slot assignment with a Hungarian-based solution; one packet per slot. | Forecast-aware per-node TDMA scheduling. | HTA-MAC uses learned trajectory features and multi-slot/sleep actions. The implemented FFSS adaptation cannot reproduce within-frame ordering in the round-level simulator and must remain labeled an adaptation. |
| Eriş et al., Sensors 2024, DOI `10.3390/s24175791` | Clustered EH-UASN; cooperative independent Q-learning/multi-armed bandits; nodes autonomously choose TDMA transmission slots based on ambient harvesting opportunities. | RL, per-node intra-cluster slot choice, EH awareness. | It is not merely 'current-energy reactive' based on inspected text; avoid that unsupported simplification. Distinguish acoustic underwater setting, stochastic piezoelectric/i.i.d. harvesting assumptions, independent learners, and slot choice versus centralized branching slot-count allocation. Never compare absolute results across these materially different environments. |
| Sarang et al., HENO-MAC, IEEE WCM 2024 / arXiv `2401.00717` | A receiver node harvests solar+wind energy and adapts duty cycle for energy-neutral operation and delay. | Hybrid harvesting and MAC duty-cycle adaptation. | Not the same clustered per-node allocation problem; no HMM trajectory-conditioned branching action is described. Its realistic trace evaluation is stronger than HTA-MAC's synthetic thermal auxiliary and should be acknowledged. |
| Tavakoli et al., AAAI 2018, DOI `10.1609/aaai.v32i1.11798` | Shared decision module with one action branch per dimension; output grows linearly with action dimensions; shared module supports coordination. | Direct architectural basis for Branching Dueling Q. | HTA-MAC adapts, rather than invents, the architecture. The independent-DQN ablation is required to test whether shared coordination helps this domain. |
| Mekathoti and Nithya, Telecommunication Systems 2025, DOI `10.1007/s11235-025-01268-0` | DRL time-slot allocation for an energy-harvesting WBAN with sleep scheduling; a two-state Markov chain is used for analytical throughput/loss. | DRL, time slots, EH, Markov modeling. | Different WBAN setting and the available abstract does not establish per-node hybrid HMM trajectory input. It narrows any broad 'RL + Markov + EH slot allocation' first claim. |
| QLDSA-MAC, JCOMSS 2026, DOI `10.24138/jcomss-2025-0154` | Q-learning-based dynamic slot assignment in WBANs. | RL-based dynamic slot allocation. | Different WBAN and no inspected hybrid-HMM harvest mechanism. Include as current adjacent work because the audit is being conducted in 2026. |

## Claims removed or narrowed

1. **Remove:** “S2A2MAC applies one HMM-derived active-period rule per cluster.”  
   **Reason:** contradicted by the primary abstract's node-level wording.
2. **Remove:** “No paper learns per-node intra-cluster TDMA allocation.”  
   **Reason:** Eriş et al. learn per-node intra-cluster slot choices.
3. **Remove:** “First hybrid-harvest per-node TDMA slot allocation.”  
   **Reason:** SHR-TDMA already optimally assigns slots per node under hybrid sources.
4. **Narrow:** “No paper explicitly models idle listening in clustered EH-WSN.”  
   **Replacement:** “The inherited HEART-CH accounting omits an explicit idle-listening term; HTA-MAC adds the term consistently to every evaluated policy.” A universal literature-absence claim requires a systematic review and full-text energy-equation audit.
5. **Narrow:** “RL-MAC uses only current energy.”  
   **Reason:** the inspected source describes learning from ambient harvesting opportunities and spatio-temporal uncertainty; the simplistic reactive-only label is not adequately supported.

## Candidate contribution language after validation

- **C1:** A bounded intra-cluster MAC formulation that conditions discrete node-level slot counts on separate solar and thermal HMM trajectory features while replaying a fixed exogenous CH schedule.
- **C2:** A Branching Dueling distributional Q architecture with output size linear in node branches, plus an explicit cluster slot-budget projection; coordination benefit is claimed only if the independent-DQN ablation supports it.
- **C3:** Explicit idle-listening accounting added to the inherited HEART-CH simulator and applied identically to every policy, with full-slot and control-header sensitivity variants.
- **C4:** Empirical characterization of the lifetime/delivery Pareto frontier across budgets 8, 12, 16, 20, and 24. Do not use “dominates” unless the registered paired results and bound validation establish it.

## Source record

- Movva et al.: https://doi.org/10.1002/dac.5202
- Gong et al. SHR-TDMA: https://doi.org/10.1049/iet-com.2019.0977
- Gong et al. FFSS/AFSS: https://doi.org/10.1049/cmu2.12243
- Eriş et al.: https://doi.org/10.3390/s24175791
- Sarang et al.: https://arxiv.org/abs/2401.00717
- Tavakoli et al.: https://doi.org/10.1609/aaai.v32i1.11798
- Mekathoti and Nithya: https://doi.org/10.1007/s11235-025-01268-0
- QLDSA-MAC: https://doi.org/10.24138/jcomss-2025-0154

## Remaining literature risk

This audit searched the supplied closest works and targeted combinations of HMM/Markov prediction, hybrid harvesting, TDMA allocation, and RL/branching architectures. It is not PRISMA-complete. Before submission, run a database-indexed search in IEEE Xplore, Scopus, Web of Science, ACM DL, and publisher databases; archive query strings, dates, inclusion criteria, and full texts used for mechanism-level claims. Until then, prefer “among the audited works” over “first” or “no published paper.”