# HTA-MAC Expanded Literature and Competitor Map (2020â€“2026)

**Search date:** 3 August 2026  
**Coverage:** peer-reviewed work published from 2020 through August 2026, with emphasis on ambient-energy-harvesting WSNs, clustered MAC, adaptive TDMA, learned sleep/transmit scheduling, harvested-energy prediction, hybrid harvesting, queue-aware control, and idle-listening accounting.

## 1. Interpretation boundary

This is an expanded targeted search, not yet a PRISMA-style systematic review. It is sufficient to correct the manuscript's competitive horizon and prioritize sources, but not sufficient by itself to prove an unrestricted â€œfirst-everâ€ claim.

The literature must be separated into three system families:

1. **Ambient EH-WSN:** nodes harvest solar, thermal, wind, vibration, or other environmental energy. This is the family closest to HTA-MAC.
2. **Wireless-powered/rechargeable sensor networks:** energy is deliberately transferred by an access point, RF source, or mobile charger. These papers support optimization methodology but are not directly comparable systems.
3. **Battery-powered or underwater learned MAC:** these support learned scheduling or scalability arguments but cannot substantiate ambient-EH superiority.

Published percentage improvements must never be compared directly with HTA-MAC unless topology, field, radio, traffic, harvesting, lifetime definition, and trial protocol are materially aligned.

## 2. Evidence-ranked direct and near-direct competitors

| Priority | Work | Topology / energy | Control and information | Relationship to HTA-MAC |
|---|---|---|---|---|
| **A1** | Gong et al., â€œSlot-hitting ratio-based TDMA schedule for hybrid energy-harvesting wireless sensor networks,â€ *IET Communications* (2020), DOI [10.1049/iet-com.2019.0977](https://doi.org/10.1049/iet-com.2019.0977) | EH-WSN; two energy-arrival classes (fixed and random) | Markov/number-theoretic slot-hitting ratios; Hungarian slot assignment; minimizes delay from awaiting a slot | **Major newly identified competitor.** It already combines hybrid energy harvesting and per-node TDMA slot assignment. HTA-MAC must differentiate solar/thermal latent-state features, learned multi-slot counts, queue/fairness/lifetime objectives, and clustered frozen-CH evaluationâ€”not claim hybrid-EH TDMA itself is new. |
| **A2** | Ge, Nan, and Guo, â€œMaximizing network throughput by cooperative reinforcement learning in clustered solar-powered wireless sensor networks,â€ *International Journal of Distributed Sensor Networks* (2021), DOI [10.1177/15501477211007411](https://doi.org/10.1177/15501477211007411) | **Clustered solar EH-WSN** | Per-node cooperative Q-learning/SARSA; residual energy, next-slot predicted harvest, and CH energy; adjusts duty cycle/transmission rate | **Major newly identified competitor.** It already learns per-node control in a clustered forecast-aware solar EH-WSN. HTA-MAC differs through dual-source state-transition features, shared branching deep value architecture, explicit cluster slot-budget projection, and frozen upstream CH schedule. |
| **A3** | Gong et al., â€œTDMA scheduling schemes targeting high channel utilization for energy-harvesting wireless sensor networks,â€ *IET Communications* (2021), DOI [10.1049/cmu2.12243](https://doi.org/10.1049/cmu2.12243) | EH-WSN; random energy and data arrivals | FFSS/AFSS estimate upcoming energy/data; Hungarian assignment and adaptive frame sizing | Strong optimization baseline. The present HTA-MAC round abstraction cannot reproduce within-frame slot order exactly, so any implementation must be labeled FFSS-style. |
| **A4** | Movva et al., â€œAn energy aware cluster-based routing and adaptive semi-synchronized MAC for energy harvesting WSN,â€ *International Journal of Communication Systems* (2022), DOI [10.1002/dac.5202](https://doi.org/10.1002/dac.5202) | Clustered EH-WSN with mobile sinks | HMM-informed SÂ²AÂ²MAC embedded with clustering and routing | Closest HMM/MAC conceptual comparison, but its full-stack assumptions differ. Do not say it has no node differentiation; contrast its semi-synchronized rule with learned per-node discrete slot counts. |
| **A5** | Sah et al., â€œTDMA policy to optimize resource utilization in Wireless Sensor Networks using reinforcement learning for ambient environment,â€ *Computer Communications* 195 (2022), DOI [10.1016/j.comcom.2022.08.013](https://doi.org/10.1016/j.comcom.2022.08.013) | Head node with multiple leaf nodes; not clearly an ambient-EH model despite title wording | MDP and Q-learning using head-node residual energy and data buffer; adjusts frame size/scheduling frequency | **Major newly identified learned-TDMA competitor.** It directly weakens any claim that RL-based TDMA frame scheduling is new. HTA-MAC operates per member under a shared cluster budget and harvest-transition state. |
| **A6** | Dutta, Bhuyan, and Biswas, â€œReinforcement learning based flow and energy management in resource-constrained wireless networks,â€ *Computer Communications* 202 (2023), DOI [10.1016/j.comcom.2023.02.011](https://doi.org/10.1016/j.comcom.2023.02.011) | Multi-hop resource-constrained WSN/IoT | Independent RL sleepâ€“listenâ€“transmit scheduling; throughput-sustainable flows, packet loss, delay, and energy | Strong learned MAC/sleep competitor. It supports the non-stationarity/scalability motivation but is not hybrid-HMM clustered allocation. |
| **A7** | Sarang et al., â€œMachine Learning Prediction Based Adaptive Duty Cycle MAC Protocol for Solar Energy Harvesting Wireless Sensor Networks,â€ *IEEE Access* (2023), DOI [10.1109/ACCESS.2023.3246108](https://doi.org/10.1109/ACCESS.2023.3246108) | Solar EH-WSN | NAR-based solar prediction; receiver duty-cycle adaptation | Essential prediction-aware MAC competitor. A faithful implementation requires receiver-initiated GreenCastalia semantics, so use as Related Work unless a separately declared exploratory baseline is built. |
| **A8** | Dutta, Bhuyan, and Biswas, â€œContextual Deep Reinforcement Learning for Flow and Energy Management in Wireless Sensor and IoT Networks,â€ *IEEE Transactions on Green Communications and Networking* (2024), DOI [10.1109/TGCN.2024.3358230](https://doi.org/10.1109/TGCN.2024.3358230) | Resource-constrained WSN/IoT | Decentralized multi-tier DRL for **joint slot allocation and transmitâ€“sleep scheduling** | **Very important target-journal competitor.** It prevents broad claims that deep RL joint slot/sleep control is absent. HTA-MAC's distinction is the clustered ambient hybrid-HMM state and branching budgeted action formulation. |
| **A9** | Sarang et al., â€œHENO-MAC: Hybrid Energy Harvesting-based Energy Neutral Operation MAC Protocol for Delay-Sensitive IoT Applications,â€ *IEEE WCNC* (2024), DOI [10.1109/WCNC57260.2024.10571258](https://doi.org/10.1109/WCNC57260.2024.10571258), [open manuscript](https://arxiv.org/abs/2401.00717) | Hybrid solarâ€“wind ambient harvesting | Energy-neutral receiver duty cycling with realistic traces | Closest hybrid-source MAC in recent IEEE literature. It is not clustered per-node TDMA allocation and does not use an HMM-conditioned branching policy. |
| **A10** | EriÅŸ, GÃ¼l, and BÃ¶lÃ¼k, â€œA Novel Medium Access Policy Based on Reinforcement Learning in Energy-Harvesting Underwater Sensor Networks,â€ *Sensors* (2024), DOI [10.3390/s24175791](https://doi.org/10.3390/s24175791) | Clustered underwater acoustic network; piezoelectric harvesting | Cooperative independent Q-learning/MAB for intra-cluster TDMA decisions | Strong learned intra-cluster scheduling competitor, but acoustic and underwater energy models prevent numerical cross-paper comparison. |
| **A11** | Farmani et al., â€œD2PG: deep deterministic policy gradient based for maximizing network throughput in clustered EH-WSN,â€ *Wireless Networks* 31 (2025; online 2024), DOI [10.1007/s11276-024-03767-5](https://doi.org/10.1007/s11276-024-03767-5) | **Clustered EH-WSN** | DDPG over continuous energy state; adapts data/transmission rate for throughput | **Major newly identified competitor.** It weakens a general claim that deep RL has not been used for clustered EH-WSN power/rate control. It does not perform hybrid-HMM, budget-constrained discrete TDMA slot allocation. |
| **A12** | Dutta, Bhuyan, and Biswas, â€œCooperative Reinforcement Learning for Energy Management in Multi-Hop Networks With Energy Harvesting,â€ *IEEE TGCN* 9(4) (2025), DOI [10.1109/TGCN.2025.3544073](https://doi.org/10.1109/TGCN.2025.3544073) | Multi-hop EH sensor/IoT networks | Two cooperative agents per node jointly learn transmit/sleep schedules; evaluates PDR and delay across solar conditions | Current target-journal state of the art for learned EH transmit/sleep control. It should be discussed prominently even though it is not clustered TDMA slot-count allocation. |
| **A13** | Hasani et al., â€œDeep reinforcement learning-based mechanism to improve the throughput of EH-WSNs,â€ *Scientific Reports* (2025), DOI [10.1038/s41598-025-14111-y](https://doi.org/10.1038/s41598-025-14111-y) | Solar EH-WSN | DQN-like deep value approximation uses continuous residual-energy state for transmission decisions | Recent direct EH-WSN DRL context. It uses current energy and explicitly ignores static energy consumption, unlike HTA-MAC's idle-aware paired environment. |
| **A14** | Nazamdin and Reid, “Safety-Constrained Reinforcement Learning for Energy-Aware Transmission Scheduling in Seismic Wireless Sensor Networks,” *Sensors* (2026), DOI [10.3390/s26113542](https://doi.org/10.3390/s26113542) | Solar-EH seismic WSN; 10–30 nodes | PPO with action masking and a runtime guard layer enforcing battery-preservation and load-balancing constraints | Current safety-aware transmission-scheduling competitor. Its explicit constraint layer is relevant to HTA-MAC's budget projection, although it does not use clustered hybrid-HMM multi-slot control. |

## 3. Important adjacent competitors and methodological precedents

### 3.1 RL, MDP, queues, and delay

1. Sharma, Mastronarde, and Chakareski, â€œDelay-Sensitive Energy-Harvesting Wireless Sensors: Optimal Scheduling, Structural Properties, and Approximation Analysis,â€ *IEEE Transactions on Communications* 68(4) (2020), DOI [10.1109/TCOMM.2019.2956510](https://doi.org/10.1109/TCOMM.2019.2956510).  
   MDP over buffer, battery, and channel states; establishes structural properties relevant to HTA-MAC's queue and energy state design.

2. Sharma, Mastronarde, and Chakareski, â€œAccelerated Structure-Aware Reinforcement Learning for Delay-Sensitive Energy Harvesting Wireless Sensors,â€ *IEEE Transactions on Signal Processing* 68 (2020), DOI [10.1109/TSP.2020.2973125](https://doi.org/10.1109/TSP.2020.2973125).  
   Strong RL scheduling precedent; useful when explaining why state structure should be exploited rather than treated as a generic black box.

3. Al-Tous and Barhumi, â€œReinforcement Learning Framework for Delay Sensitive Energy Harvesting Wireless Sensor Networks,â€ *IEEE Sensors Journal* 21(5) (2021), DOI [10.1109/JSEN.2020.3044049](https://doi.org/10.1109/JSEN.2020.3044049).  
   Centralized and distributed SARSA in multi-hop EH-WSNs; directly relevant to the independent-learner versus centralized/shared-policy discussion.

4. Zhao and Zhao, â€œDeep Reinforcement Learning Resource Allocation in Wireless Sensor Networks With Energy Harvesting and Relay,â€ *IEEE Internet of Things Journal* 9(3) (2022), DOI [10.1109/JIOT.2021.3094465](https://doi.org/10.1109/JIOT.2021.3094465).  
   Actorâ€“critic power/time allocation under causal battery and channel state. It is resource allocation rather than clustered MAC, but is a major deep-RL EH-WSN precedent.

5. Han and Gong, â€œStatus update control based on reinforcement learning in energy harvesting sensor networks,â€ *Frontiers in Communications and Networks* (2022), DOI [10.3389/frcmn.2022.933047](https://doi.org/10.3389/frcmn.2022.933047).  
   Q-learning/DQN scheduling under energy and channel dynamics; supports a freshness-aware interpretation of queue TTL and stale-packet loss.

6. Jin et al., â€œDeep reinforcement learning based scheduling for minimizing age of information in wireless powered sensor networks,â€ *Computer Communications* 191 (2022), DOI [10.1016/j.comcom.2022.04.007](https://doi.org/10.1016/j.comcom.2022.04.007).  
   Wireless-powered rather than ambient-EH; relevant to DRL/AoI scheduling methodology only.

7. â€œLearning to Transmit Fresh Information in Energy Harvesting Networks,â€ *IEEE TGCN* 6(4) (2022), DOI [10.1109/TGCN.2022.3190007](https://doi.org/10.1109/TGCN.2022.3190007).  
   Supervised and actorâ€“critic methods for EH scheduling and power allocation with an AoI objective.

8. Mohammadi and Shirmohammadi, â€œRLSÂ²: An energy efficient reinforcement learning-based sleep scheduling for energy harvesting WBANs,â€ *Computer Networks* 229 (2023), DOI [10.1016/j.comnet.2023.109781](https://doi.org/10.1016/j.comnet.2023.109781).  
   Per-node sleep/wake schedules and action masking under heterogeneous harvesting; body-network topology makes it adjacent, not directly comparable.

9. â€œDRDC: Deep reinforcement learning based duty cycle for energy harvesting body sensor node,â€ *Energy Reports* (2023), DOI [10.1016/j.egyr.2022.12.138](https://doi.org/10.1016/j.egyr.2022.12.138).  
   DQN selects duty cycle using residual energy, harvestable light, sensed-data change, and sleep behavior.

10. Seifullaev et al., â€œReinforcement Learning-Based Transmission Policies for Energy Harvesting Powered Sensors,â€ *IEEE TGCN* 8(4) (2024), DOI [10.1109/TGCN.2024.3374899](https://doi.org/10.1109/TGCN.2024.3374899).  
    First-order Markov harvest model, Bayesian filtering/smoothing, scenario-change detection, and switched RL transmission policies. This is highly relevant to claims about using predicted latent harvest dynamics.

11. “Adaptive Micro-sleep Scheduling for Batteryless IoT Sensors using Energy-Harvesting-Aware Reinforcement Learning,” *IEEE ICSEDIS* (2026), DOI [10.1109/ICSEDIS68157.2026.11518471](https://doi.org/10.1109/ICSEDIS68157.2026.11518471).  
    Current batteryless ambient-EH sleep-scheduling work; useful recent context, but not clustered TDMA.

12. “A Bilevel Deep Learning Optimization Framework for Joint Energy Harvesting Prediction and Energy-Aware Scheduling in IoT-Based Wireless Sensor Networks,” *Computers, Materials & Continua* (2026), DOI [10.32604/cmc.2026.079984](https://doi.org/10.32604/cmc.2026.079984).  
    Very recent joint prediction/scheduling work. It controls duty cycle, sensing rate, and transmission power with constrained horizon-based optimization rather than per-node TDMA slot counts. Because of its recency, its implementation and evaluation details require full-text audit before strong comparison.

## 4. Rule-based and analytical MAC comparators

1. Kaur, Singh, and Sohi, â€œAdaptive MAC Protocol for Solar Energy Harvesting Based Wireless Sensor Networks in Agriculture,â€ *Wireless Personal Communications* (2020), DOI [10.1007/s11277-019-06985-9](https://doi.org/10.1007/s11277-019-06985-9).

2. Khan et al., â€œAn efficient medium access control protocol for RF energy harvesting based IoT devices,â€ *Computer Communications* (2021), DOI [10.1016/j.comcom.2021.02.011](https://doi.org/10.1016/j.comcom.2021.02.011). E-MAC-IoT adapts IEEE 802.15.4 duty cycle/GTS admission using residual and accumulated RF energy.

3. â€œResidual Energy Estimation-Based MAC Protocol for Wireless Powered Sensor Networks,â€ *Sensors* (2021), DOI [10.3390/s21227617](https://doi.org/10.3390/s21227617). REE-MAC is wireless-powered, not ambient-harvesting, but is useful for contrasting residual-energy estimation with harvest-trajectory conditioning.

4. Sarang et al., â€œEnergy Neutral Operation based Adaptive Duty Cycle MAC Protocol for Solar Energy Harvesting Wireless Sensor Networks,â€ *IEEE VTC-Spring* (2022), DOI [10.1109/VTC2022-Spring54318.2022.9860635](https://doi.org/10.1109/VTC2022-Spring54318.2022.9860635). ENCODMAC is a rule-based solar energy-neutral precursor to PADC/HENO-MAC.

5. Blondia, â€œEvaluation of the end-to-end response times in an energy harvesting wireless sensor network using a receiver-initiated MAC protocol,â€ *Ad Hoc Networks* (2022), DOI [10.1016/j.adhoc.2022.102971](https://doi.org/10.1016/j.adhoc.2022.102971). Provides Markov-chain response-time analysis.

6. â€œETI-MAC: An Energy-Harvested Transmitter-Initiated MAC Protocol for Wireless Sensor Networks,â€ *Applied Computer Systems* (2023/2024 issue), DOI [10.2478/acss-2023-0021](https://doi.org/10.2478/acss-2023-0021). Uses harvested-energy rate to select sleep duration under low-power listening.

7. Kaur and Singh, â€œAdaptive Data Transmission Protocols for Energy Harvesting WSNs Used in Agriculture,â€ *Journal of Telecommunications and Information Technology* (2024), DOI [10.26636/jtit.2024.1.1390](https://doi.org/10.26636/jtit.2024.1.1390). Introduces receiver-initiated solar SHMAC and highlights collision/duty-cycle trade-offs.

## 5. Harvest prediction and hybrid-source context

1. Sah et al., â€œHarvested Energy Prediction Technique for Solar-Powered Wireless Sensor Networks,â€ *IEEE Sensors Journal* 23 (2023), DOI [10.1109/JSEN.2022.3208730](https://doi.org/10.1109/JSEN.2022.3208730).

2. Frimane, Munkhammar, and van der Meer, â€œInfinite hidden Markov model for short-term solar irradiance forecasting,â€ *Solar Energy* 244 (2022), DOI [10.1016/j.solener.2022.08.041](https://doi.org/10.1016/j.solener.2022.08.041).

3. â€œEnergy Prediction for Energy-Harvesting Wireless Sensor: A Systematic Mapping Study,â€ *Electronics* 12(20) (2023), DOI [10.3390/electronics12204304](https://doi.org/10.3390/electronics12204304).

4. Bao et al., â€œDistributed dynamic scheduling algorithm of target coverage for wireless sensor networks with hybrid energy harvesting system,â€ *Scientific Reports* (2024), DOI [10.1038/s41598-024-78671-1](https://doi.org/10.1038/s41598-024-78671-1). Hybrid harvesting and time-slot scheduling are present, but the controlled object is target coverage rather than intra-cluster MAC.

5. Nicot et al., â€œAn Autonomous Wireless Sensor Node Based on Hybrid RF Solar Energy Harvesting,â€ *Wireless Power Transfer* (online 2024), DOI [10.1155/2021/6642938](https://doi.org/10.1155/2021/6642938). Hardware/system evidence for hybrid-source feasibility; not a scheduling competitor.

6. Naifar, Kanoun, and Trigona, â€œEnergy Harvesting Technologies and Applications for the Internet of Things and Wireless Sensor Networks,â€ *Sensors* (2024), DOI [10.3390/s24144688](https://doi.org/10.3390/s24144688). Broad source-technology context spanning solar, thermal, vibration, and RF.

## 6. Idle listening and radio-energy evidence

1. Salam et al., â€œEnergy-Efficient Method for Wireless Sensor Networks Low-Power Radio Operation in Internet of Things,â€ *Electronics* (2020), DOI [10.3390/electronics9020320](https://doi.org/10.3390/electronics9020320).

2. Barroca et al., â€œPerformance enhancement of IEEE 802.15.4 by employing RTS/CTS and frame concatenation,â€ *IET Wireless Sensor Systems* (2020), DOI [10.1049/iet-wss.2019.0003](https://doi.org/10.1049/iet-wss.2019.0003).

3. â€œAdaptive clear channel assessment (A-CCA): Energy efficient method to improve wireless sensor networks operations,â€ *AEÃœ* (2021), DOI [10.1016/j.aeue.2020.153603](https://doi.org/10.1016/j.aeue.2020.153603).

4. â€œEnergy-Aware QoS MAC Protocol Based on Prioritized-Data and Multi-Hop Routing for Wireless Sensor Networks,â€ *Sensors* (2022), DOI [10.3390/s22072598](https://doi.org/10.3390/s22072598). Its CC2420/TelosB table reports equal receive and idle power, supporting an approximation but not HTA-MAC's exact slot duration.

These sources support the importance of idle listening. They do **not** support the claim that no EH-WSN paper has ever modeled idle energy.

## 7. Surveys and taxonomy anchors

1. â€œA Comprehensive Review on Energy Harvesting Integration in IoT Systems from MAC Layer Perspective: Challenges and Opportunities,â€ *Sensors* (2021), DOI [10.3390/s21093097](https://doi.org/10.3390/s21093097).

2. Sandhu et al., â€œTask Scheduling for Energy-Harvesting-Based IoT: A Survey and Critical Analysis,â€ *IEEE Internet of Things Journal* (2021), DOI [10.1109/JIOT.2021.3086186](https://doi.org/10.1109/JIOT.2021.3086186).

3. â€œReinforcement Learning TDMA-Based MAC Scheduling in the Industrial Internet of Things: A Survey,â€ *IFAC-PapersOnLine* (2022), DOI [10.1016/j.ifacol.2022.08.014](https://doi.org/10.1016/j.ifacol.2022.08.014).

4. â€œEnergy harvesting techniques for wireless sensor networks: A systematic literature review,â€ *Sustainable Energy Technologies and Assessments* (2024/2025), [publisher record](https://www.sciencedirect.com/science/article/pii/S2211467X24003262).

Do not use the retracted 2022 underwater MAC/routing survey previously identified.

## 8. Corrected novelty horizon

### What the literature already contains

- Hybrid-energy-aware TDMA slot assignment: SHR-TDMA (2020).
- Per-node cooperative RL in clustered, solar-powered EH-WSNs using next-slot predicted energy: Ge et al. (2021).
- Forecast-aware TDMA assignment and adaptive frame sizing: FFSS/AFSS (2021).
- HMM-informed EH-WSN MAC integrated with clustering/routing: SÂ²AÂ²MAC (2022).
- Q-learning-based TDMA frame scheduling: Sah et al. (2022).
- Deep RL for joint slot allocation and transmit/sleep control: Dutta et al. (2024).
- Deep continuous-control power/rate management in clustered EH-WSNs: D2PG (2025 issue).
- Hybrid-source energy-neutral MAC using realistic traces: HENO-MAC (2024).
- Safety-constrained solar-EH transmission scheduling with PPO, action masking, and a guard layer: Nazamdin and Reid (2026).
- Joint learned energy prediction and constrained energy-aware WSN scheduling: the 2026 bilevel framework.

### Narrow claim potentially supportable after systematic verification

> To the best of our documented search, HTA-MAC is the first evaluated clustered terrestrial EH-WSN scheduler to combine per-node solar-and-thermal HMM state-transition features with budget-constrained discrete multi-slot allocation through a shared Branching Dueling value network under a frozen exogenous CH schedule.

This sentence is intentionally conjunctive. Removing any of the qualifiers makes it vulnerable to the papers above. Replace â€œposteriorâ€ with â€œstate-transition probability vectorâ€ unless a true Bayesian forward-filter posterior is actually reused from HEART-CH.

## 9. Consequences for the four planned contributions

- **C1 needs narrowing.** Per-node forecast-aware control in clustered solar EH-WSNs already exists, and hybrid-EH TDMA already exists. The contribution is the specific solar/thermal HMM-transition-conditioned **multi-slot budget allocation** formulation.
- **C2 remains plausible as an architectural contribution.** Phrase it as applying Branching Dueling C51 to the constrained per-member slot-count vector, not as inventing scalable learned MAC generally.
- **C3 must be reframed.** Claim explicit idle-listening accounting relative to HEART-CH and the shared paired environment; do not claim universal absence from EH-WSN literature.
- **C4 must be conditional.** A strict dominance theorem requires stated assumptions and empirical validation. It cannot be advertised as general dominance over all static and threshold methods before Phase 5 evidence.

## 10. Baseline implications

The frozen confirmatory baseline family should not be changed silently after training begins. However:

1. Keep SÂ²AÂ²MAC-style and FFSS-style as declared confirmatory adaptations.
2. Add **SHR-TDMA**, Ge et al.'s cooperative RL power manager, Sah et al.'s Q-learning TDMA scheduler, and D2PG to the qualitative mechanism-comparison table.
3. If implementation time permits, implement SHR-TDMA and the Ge et al. policy only as **preregistered exploratory baselines**, clearly separated from the already frozen confirmatory analysis.
4. Discuss Contextual DRL (2024) and Cooperative RL (2025) prominently because both appear in the target journal and are stronger state-of-the-art positioning references than generic underwater or WBAN work.

## 11. Recommended Related Work structure

1. **Analytical and forecast-aware EH-TDMA:** SHR-TDMA; FFSS/AFSS.
2. **HMM and predictive EH-MAC:** SÂ²AÂ²MAC; PADC-MAC; ENCODMAC; HENO-MAC.
3. **Learned clustered and EH scheduling:** Ge et al.; Sah et al.; Zhao and Zhao; D2PG; EH-UASN.
4. **Learned slot/sleep control and target-journal frontier:** Dutta et al. 2023/2024/2025; Seifullaev et al. 2024.
5. **Accounting mechanism:** CC2420 idle/listen literature, followed by the precise statement that HEART-CH omitted this term.

This structure makes the distinction cumulative and testable instead of relying on one broad novelty sentence.

