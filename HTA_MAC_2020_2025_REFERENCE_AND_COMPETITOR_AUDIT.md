# HTA-MAC 2020–2025 Reference and Competitor Audit

**Search date:** 3 August 2026  
**Scope:** Peer-reviewed journal and conference papers published from 2020 through 2025 that are relevant to energy-harvesting MAC, TDMA scheduling, learned duty cycling, harvested-energy prediction, HMM forecasting, idle-listening accounting, and scalable learned scheduling.

This is a targeted literature audit, not yet a systematic-review proof of a universal “first” claim. DOI/publisher links were verified before inclusion.

## 1. Closest competitors

| Year | Paper | Mechanism and relationship to HTA-MAC | Use in our paper |
|---|---|---|---|
| 2022 | Movva, Kamarajugadda, and Polipalli, “An energy aware cluster-based routing and adaptive semi-synchronized MAC for energy harvesting WSN,” *International Journal of Communication Systems*, DOI [10.1002/dac.5202](https://doi.org/10.1002/dac.5202) | Clustered EH-WSN; HMM-driven S²A²MAC active-period scheduling embedded in a larger clustering/routing system. This is the closest HMM/MAC conceptual competitor. | Keep the current declared S²A²MAC-style adaptation, but describe it as an adaptation because its mobile sinks, clustering, routing, and NS-3 setup differ. |
| 2021 | Gong et al., “TDMA scheduling schemes targeting high channel utilization for energy-harvesting wireless sensor networks,” *IET Communications*, DOI [10.1049/cmu2.12243](https://doi.org/10.1049/cmu2.12243) | FFSS/AFSS use upcoming energy and data to optimize slot assignment; the fixed-frame problem is solved with a Hungarian-based method. | Strong direct scheduling baseline. Our FFSS adaptation must disclose that the current round-level environment cannot represent within-frame slot order. |
| 2023 | Sarang et al., “Machine Learning Prediction Based Adaptive Duty Cycle MAC Protocol for Solar Energy Harvesting Wireless Sensor Networks,” *IEEE Access*, DOI [10.1109/ACCESS.2023.3246108](https://doi.org/10.1109/ACCESS.2023.3246108) | PADC-MAC predicts future solar intake with a nonlinear autoregressive neural network and adapts receiver duty cycle. | Essential Related Work competitor for prediction-aware MAC. A faithful new baseline would require GreenCastalia-style receiver-initiated semantics and should not be added post hoc to the frozen confirmatory family. |
| 2024 | Sarang et al., “HENO-MAC: Hybrid Energy Harvesting-based Energy Neutral Operation MAC Protocol for Delay-Sensitive IoT Applications,” *IEEE WCNC 2024*, DOI [10.1109/WCNC57260.2024.10571258](https://doi.org/10.1109/WCNC57260.2024.10571258), [open manuscript](https://arxiv.org/abs/2401.00717) | Hybrid solar–wind harvesting, energy-neutral operation, and adaptive duty cycling using realistic traces. | Closest hybrid-source MAC competitor. It is not solar–thermal, not clustered per-node slot allocation, and not an RL slot-count policy. |
| 2024 | Eris, Gül, and Bölük, “A Novel Medium Access Policy Based on Reinforcement Learning in Energy-Harvesting Underwater Sensor Networks,” *Sensors*, DOI [10.3390/s24175791](https://doi.org/10.3390/s24175791) | Clustered intra-cluster TDMA with cooperative independent Q-learning/multi-armed-bandit decisions under stochastic piezoelectric harvesting. | Strongest recent learned-slot competitor. Use in Related Work, but do not compare its published percentages numerically with HTA-MAC because acoustic energy/channel assumptions are materially different. |
| 2021 | Khan et al., “An efficient medium access control protocol for RF energy harvesting based IoT devices,” *Computer Communications*, DOI [10.1016/j.comcom.2021.02.011](https://doi.org/10.1016/j.comcom.2021.02.011) | E-MAC-IoT adapts IEEE 802.15.4 duty cycle and GTS admission using residual and accumulated RF-harvested energy. | Direct energy-aware GTS/MAC comparator in Related Work; different single-hop RF-harvesting system prevents cross-paper numerical claims. |
| 2022 | Sarang et al., “Energy Neutral Operation based Adaptive Duty Cycle MAC Protocol for Solar Energy Harvesting Wireless Sensor Networks,” *IEEE VTC-Spring 2022*, DOI [10.1109/VTC2022-Spring54318.2022.9860635](https://doi.org/10.1109/VTC2022-Spring54318.2022.9860635) | ENCODMAC adjusts duty cycle from battery level and surplus solar energy to operate near energy neutrality. | Relevant rule-based prediction/ENO comparator; cite as precursor to PADC-MAC and HENO-MAC. |
| 2021 | Gao, Zhang, and Zhang, “HAS-MAC: A Hybrid Asynchronous and Synchronous Communication System for Energy-Harvesting Wireless Sensor Networks,” *Wireless Personal Communications*, DOI [10.1007/s11277-021-08304-7](https://doi.org/10.1007/s11277-021-08304-7) | Hybrid asynchronous/synchronous MAC for EH-WSNs. | Related Work comparator for hybrid MAC operation; “hybrid” here describes communication modes, not two harvesting sources. |
| 2021 | Charoenchaiprakit, Piyarat, and Woradit, “Optimal Data Transfer of SEH-WSN Node via MDP Based on Duty Cycle and Battery Energy,” *IEEE Access*, DOI [10.1109/ACCESS.2021.3086883](https://doi.org/10.1109/ACCESS.2021.3086883) | MDP-based solar-EH duty-cycle/data-transfer control using energy state. | Useful non-deep-RL scheduling comparator and analytical precedent. |
| 2020 | Kaur, Singh, and Sohi, “Adaptive MAC Protocol for Solar Energy Harvesting Based Wireless Sensor Networks in Agriculture,” *Wireless Personal Communications*, DOI [10.1007/s11277-019-06985-9](https://doi.org/10.1007/s11277-019-06985-9) | Weather-adaptive, receiver-initiated multilayer solar-EH MAC evaluated for agriculture. | Earliest paper in this 2020–2025 set; cite as a recent rule-based solar-EH MAC. |
| 2023 | “DRDC: Deep reinforcement learning based duty cycle for energy harvesting body sensor node,” *Energy Reports*, DOI [10.1016/j.egyr.2022.12.138](https://doi.org/10.1016/j.egyr.2022.12.138) | DQN selects duty cycle using energy, harvestable light, sensed-data change, and sleep behavior for an EH body sensor. | Learned duty-cycle competitor; different single-node/WBAN topology means Related Work only. |
| 2025 | Jiang et al., “Underwater Acoustic MAC Protocol for Multi-Objective Optimization Based on Multi-Agent Reinforcement Learning,” *Drones*, DOI [10.3390/drones9020123](https://doi.org/10.3390/drones9020123) | MOMA-MAC uses MAPPO with throughput, energy-efficiency, and fairness objectives. | Recent learned-MAC context. It is underwater and not an ambient-EH trajectory scheduler. |

## 2. Scheduling and learned-access papers supporting the formulation

1. Dutta and Biswas, “Distributed Reinforcement Learning for scalable wireless medium access in IoTs and sensor networks,” *Computer Networks* 202 (2022), DOI [10.1016/j.comnet.2021.108662](https://doi.org/10.1016/j.comnet.2021.108662).  
   Relevant to the scalability/non-stationarity discussion around distributed independent MAC learners.

2. Jin et al., “Deep reinforcement learning based scheduling for minimizing age of information in wireless powered sensor networks,” *Computer Communications* 191 (2022), DOI [10.1016/j.comcom.2022.04.007](https://doi.org/10.1016/j.comcom.2022.04.007).  
   Combines DRL and Lyapunov optimization for energy/channel/AoI-aware transmission decisions.

3. Hribar et al., “Timely and sustainable: Utilising correlation in status updates of battery-powered and energy-harvesting sensors using Deep Reinforcement Learning,” *Computer Communications* 192 (2022), DOI [10.1016/j.comcom.2022.05.030](https://doi.org/10.1016/j.comcom.2022.05.030).  
   Learns per-sensor update intervals while balancing information freshness, energy, and lifetime.

4. “Learning to Transmit Fresh Information in Energy Harvesting Networks,” *IEEE Transactions on Green Communications and Networking* 6(4) (2022), DOI [10.1109/TGCN.2022.3190007](https://doi.org/10.1109/TGCN.2022.3190007).  
   Especially relevant to the target journal and the lifetime/freshness trade-off; uses supervised and actor–critic solutions for EH scheduling and power allocation.

5. Blondia, “Evaluation of the end-to-end response times in an energy harvesting wireless sensor network using a receiver-initiated MAC protocol,” *Ad Hoc Networks* (2022), DOI [10.1016/j.adhoc.2022.102971](https://doi.org/10.1016/j.adhoc.2022.102971).  
   Provides queueing/Markov-chain analysis for EH receiver-initiated MAC delay.

6. “Scheduling recurring and dependent tasks in EH-WSNs,” *Sustainable Computing: Informatics and Systems* 27 (2020), DOI [10.1016/j.suscom.2020.100409](https://doi.org/10.1016/j.suscom.2020.100409).  
   Uses real harvesting traces and energy-neutral scheduling; supports the broader prediction-aware scheduling motivation.

## 3. Harvest forecasting and HMM references

1. Sah et al., “Harvested Energy Prediction Technique for Solar-Powered Wireless Sensor Networks,” *IEEE Sensors Journal* 23 (2023), DOI [10.1109/JSEN.2022.3208730](https://doi.org/10.1109/JSEN.2022.3208730).  
   Modified profile-based solar prediction; useful for positioning HMM trajectory features against non-HMM energy predictors.

2. Frimane, Munkhammar, and van der Meer, “Infinite hidden Markov model for short-term solar irradiance forecasting,” *Solar Energy* 244 (2022), DOI [10.1016/j.solener.2022.08.041](https://doi.org/10.1016/j.solener.2022.08.041).  
   Direct support for latent-state HMM solar forecasting, while operating outside WSN/MAC control.

3. “Energy Prediction for Energy-Harvesting Wireless Sensor: A Systematic Mapping Study,” *Electronics* 12(20) (2023), DOI [10.3390/electronics12204304](https://doi.org/10.3390/electronics12204304).  
   Use to organize prediction-model families and avoid claiming HMM is the only forecasting approach.

4. Sandhu et al., “Task Scheduling for Energy-Harvesting-Based IoT: A Survey and Critical Analysis,” *IEEE Internet of Things Journal* (2021), DOI [10.1109/JIOT.2021.3086186](https://doi.org/10.1109/JIOT.2021.3086186).  
   Broad scheduling taxonomy spanning solar, kinetic, thermal, and RF harvesting.

5. Han and Gong, “Status update control based on reinforcement learning in energy harvesting sensor networks,” *Frontiers in Communications and Networks* (2022), DOI [10.3389/frcmn.2022.933047](https://doi.org/10.3389/frcmn.2022.933047).  
   RL scheduling that balances information error and energy cost under channel-state dynamics.

## 4. Idle-listening and radio-energy evidence

1. Salam et al., “Energy-Efficient Method for Wireless Sensor Networks Low-Power Radio Operation in Internet of Things,” *Electronics* 9(2) (2020), DOI [10.3390/electronics9020320](https://doi.org/10.3390/electronics9020320).  
   Analyzes false wakeups and idle listening with CC2420/ContikiMAC-style radio operation.

2. “Adaptive clear channel assessment (A-CCA): Energy efficient method to improve wireless sensor networks operations,” *AEÜ—International Journal of Electronics and Communications* 131 (2021), DOI [10.1016/j.aeue.2020.153603](https://doi.org/10.1016/j.aeue.2020.153603).  
   Hardware/simulation evidence that reducing idle-listening and false-wakeup time reduces energy consumption.

3. “Energy-Aware QoS MAC Protocol Based on Prioritized-Data and Multi-Hop Routing for Wireless Sensor Networks,” *Sensors* 22(7) (2022), DOI [10.3390/s22072598](https://doi.org/10.3390/s22072598).  
   Uses CC2420/TelosB parameters where receive and idle operation have the same reported power, supporting the chosen approximation while not validating the exact HTA-MAC slot-duration assumption.

4. Barroca et al., “Performance enhancement of IEEE 802.15.4 by employing RTS/CTS and frame concatenation,” *IET Wireless Sensor Systems* (2020), DOI [10.1049/iet-wss.2019.0003](https://doi.org/10.1049/iet-wss.2019.0003).  
   Separates transmit, receive/listen, sleep, and wasted energy using CC2420 parameters.

## 5. Recommended use in the HTA-MAC manuscript

### Existing confirmatory/implemented competitors

Keep the preregistered family unchanged:

- static equal TDMA;
- energy-proportional allocation;
- harvest-proportional allocation;
- S²A²MAC-style adaptation;
- FFSS-style adaptation;
- random-budgeted diagnostic;
- the five HTA-MAC budget arms.

Do not add PADC-MAC, HENO-MAC, ENCODMAC, or E-MAC-IoT to the frozen confirmatory family after training/evaluation has begun. They can be:

- discussed in Related Work;
- placed in a qualitative mechanism-comparison table;
- implemented later as explicitly exploratory baselines or in a preregistered follow-up evaluation.

### Safest differentiation statements

- HTA-MAC learns discrete per-node slot counts inside a clustered terrestrial EH-WSN from residual energy, queue state, and hybrid solar/thermal state-transition features.
- FFSS/AFSS are optimization-based and forecast upcoming energy/data, but do not learn a Branching-DQN policy.
- PADC-MAC predicts solar energy and adapts receiver duty cycle rather than allocating per-node clustered slot counts.
- HENO-MAC handles hybrid solar/wind harvesting and energy-neutral duty cycling, rather than per-node hybrid-HMM TDMA slot allocation.
- The 2024 EH-UASN paper learns intra-cluster transmission timing, but uses underwater acoustic/piezoelectric assumptions and cooperative independent Q-learning rather than the terrestrial frozen-CH Branching-DQN formulation.

### Claims that should not be used without a broader systematic review

- “No published EH-WSN paper models idle listening.”
- “S²A²MAC cannot differentiate nodes at all.”
- “HTA-MAC is the first learned harvest-aware MAC.”
- “No paper combines HMM and MAC scheduling.”

A narrower claim may ultimately be defensible—per-node discrete slot-count allocation conditioned on hybrid solar/thermal state-transition features through a shared branching value architecture—but it still requires a systematic database search and documented inclusion/exclusion protocol.

## 6. Explicit exclusion

The 2022 article titled “A Comprehensive Survey of Energy-Efficient MAC and Routing Protocols for Underwater Wireless Sensor Networks” is marked retracted by its publisher and must not be cited as supporting evidence.

