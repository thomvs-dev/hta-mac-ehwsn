# HTA-MAC Related-Paper Performance Comparison

**Prepared:** 6 August 2026  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Evidence status:** development-only; no held-out or publication-level superiority claim

## 1. Bottom-line assessment

HTA-MAC currently shows a real and useful strength, but not universal superiority.

- The trained **B12 seed-4299** checkpoint is an **energy/lifetime-first policy**. Against the identical-simulator static baseline it raises median FND by **62.60%**, throughput by **29.76%**, and packets/J by **62.28%**, while reducing idle energy by **30.22%**.
- That gain is purchased with weaker service: delivery is **18.80% lower** and stale-drop ratio is **40.57% higher** than static. Only **1/5** development schedules passes all three QoS constraints.
- The **B16 inference diagnostic** is the most balanced observed operating point: it passes all QoS thresholds on **5/5** development schedules and still exceeds static in FND, throughput, and packets/J. It is not yet a valid trained B16 result because the checkpoint was trained at B12.
- Related papers report throughput improvements of roughly **5.7-30.1%**, delay reductions up to **28.5%**, and learned-scheduling lifetime improvements ranging from single digits to tens of percent. HTA-MAC's internal effect sizes are competitive in magnitude, but the studies do not share enough assumptions for a direct leaderboard.
- The main scientific risk is therefore not that HTA-MAC is obviously weak. It is that the present B12 policy optimizes the wrong point on the lifetime-versus-service frontier, and that a raw cross-paper comparison would overstate the evidence.

## 2. Retrieval and verification boundary

The requested Firecrawl scrape workflow was invoked first. The installed Firecrawl CLI was not authenticated in this environment, so new Firecrawl API retrieval could not proceed. Existing Firecrawl research artifacts in `.firecrawl/hta_mac_performance_research_20260804/` were used for discovery, and every quantitative statement below was rechecked against a primary publisher page, open full text, or author manuscript where available.

The comparison follows three evidence tiers:

1. **Tier A — numerical and causal:** policies executed inside the current HTA-MAC simulator on the same development schedules.
2. **Tier B — normalized contextual:** a paper's method compared with that paper's own baseline. Effect sizes can be discussed, but not ranked directly against ours.
3. **Tier C — mechanism only:** closely related work whose published metric or system model does not map to HTA-MAC.

## 3. What exactly was evaluated in our framework

| Item | HTA-MAC setting |
|---|---|
| Network | 100 nodes, 100 m x 100 m field, BS at (50, 175) m |
| Initial energy | 0.5 J/node |
| Traffic | one 4000-bit data packet per alive node per round |
| Clustering | frozen exogenous HEART-CH schedule; MAC intervention only |
| Mobility | 20% mobile nodes, random waypoint |
| Harvesting | eight-state solar plus four-state synthetic-auxiliary thermal HMM |
| Queue | maximum 5 packets; TTL 3 rounds |
| Action | per-node 0-3 slots under a hard cluster budget |
| Energy accounting | transmit, receive/aggregation, and explicit idle-listening energy |
| Evaluation | development seeds 2300-2304, common horizon 300 rounds |
| Trained candidate | seed 4299, trained at budget 12 |

The thermal component is not a field-trained thermal model. The current results must therefore be described as hybrid-source simulator evidence, not real solar-thermal deployment validation.

## 4. Our measured results

### 4.1 Absolute development medians

| Policy | FND | HND | Throughput | Delivery | Stale | Fairness | Idle J | Packets/J | Joint QoS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HTA-MAC B12, trained** | **213** | censored at 300 | **15,919** | 0.53672 | 0.45128 | 0.97854 | **31.1618** | **377.12** | 1/5 |
| **HTA-MAC B16, inference diagnostic** | **169** | 263 | **16,891** | 0.64045 | 0.34775 | **0.98682** | 42.4104 | 315.33 | **5/5** |
| Static equal | 131 | not primary here | 12,268 | 0.66095 | 0.32104 | 0.98119 | 44.6596 | 232.38 | reference |
| Energy proportional | 129 | not primary here | 12,041 | 0.79495 | 0.19183 | 0.99121 | 44.9928 | 228.40 | reference |
| Harvest proportional | 124 | not primary here | 14,479 | 0.66951 | 0.32307 | 0.98644 | 43.8336 | 271.51 | reference |
| FFSS-adapted | 143 | not primary here | 12,049 | 0.65452 | 0.32739 | 0.98655 | 45.0424 | 227.59 | reference |
| S2A2MAC-adapted | 206 | not primary here | 13,048 | 0.47589 | 0.51280 | 0.90502 | 31.7396 | 329.77 | reference |

`B16` is a counterfactual inference-budget diagnostic using the B12-trained weights. It cannot be called the final B16 model.

### 4.2 Normalized internal comparison

Positive is an increase in the named metric; for stale ratio and idle energy, a negative change is desirable.

| Comparison | FND | Throughput | Delivery | Stale | Fairness | Idle energy | Packets/J |
|---|---:|---:|---:|---:|---:|---:|---:|
| B12 vs static | +62.60% | +29.76% | -18.80% | +40.57% | -0.27% | -30.22% | +62.28% |
| B12 vs S2A2-adapted | +3.40% | +22.00% | +12.78% | -12.00% | +8.12% | -1.82% | +14.36% |
| B16 diagnostic vs static | +29.01% | +37.68% | -3.10% | +8.32% | +0.57% | -5.04% | +35.69% |
| B16 diagnostic vs S2A2-adapted | -17.96% | +29.45% | +34.58% | -32.19% | +9.04% | +33.62% | -4.38% |

Interpretation:

- B12 dominates the adapted S2A2 policy across all listed metrics in this simulator.
- B12 does not dominate the service-oriented baselines. It preserves nodes and energy partly by serving less traffic before expiry.
- B16 is the best observed compromise, but the increased slot capacity raises idle energy and advances node deaths relative to B12.

## 5. Paper-by-paper comparison

### 5.1 Ge, Nan, and Guo (2021): cooperative RL in clustered solar EH-WSNs

Primary full text: [International Journal of Distributed Sensor Networks paper](https://journals.sagepub.com/doi/pdf/10.1177/15501477211007411)

This is one of the closest system-level papers. It uses clustered solar-powered nodes, cooperative Q-learning/SARSA, predicted next-slot solar energy, residual node energy, CH energy, and learned transmission-rate/duty-cycle control.

Key setup differences:

- 20, 40, or 60 nodes in a 200 m x 200 m field with the BS at the center;
- 0.5 J initial energy and 4000-bit packets, which match two of our radio-level constants;
- rotating/optimized clustering is part of the intervention;
- one episode represents 24 hours and 240 rounds;
- real hourly solar traces are used;
- static energy consumption and battery self-consumption are set to zero.

Reported results:

- learned throughput oscillates around 170,000-218,000 packets after training for the 60-node case;
- the best learned result described near the static comparison is about 218,000 packets versus 213,000 for the tuned static rate, approximately **+2.35%** for that configuration;
- under battery degradation, learned throughput is **16.6-30.1%** above static and accumulated dead-node events are **21.0-45.9%** lower;
- under equipment-power changes, throughput gains are **5.5-18.5%** and dead-node-event reductions are **8.9-37.3%**.

Comparison verdict: HTA-MAC B12's +29.76% throughput over static and +62.60% FND are competitive in magnitude, but they are not directly greater results. Ge et al. optimize clustering and transmission rate together, use a center BS and real solar traces, omit static/idle consumption, and count accumulated dead-node events rather than our first-death time.

### 5.2 Gong et al. (2020): SHR-TDMA for hybrid energy arrivals

Primary source: [IET Communications paper](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/iet-com.2019.0977)

SHR-TDMA models hybrid fixed-inter-arrival and random-inter-arrival energy sources, derives slot-hitting ratios, and solves a Hungarian assignment minimizing delay caused by waiting for a slot.

Reported result: the paper reports significant DAFAS improvement over existing TDMA schemes, but the accessible primary text does not provide one universal percentage suitable for extraction.

Comparison verdict: mechanism-level direct relevance, numerical non-comparability. It assigns one fixed position per node in a frame and optimizes within-frame waiting delay. HTA-MAC allocates 0-3 slot counts under a cluster budget and measures lifetime, service, fairness, stale expiry, and energy efficiency at round level.

### 5.3 Gong et al. (2021): FFSS/AFSS channel-utilization scheduling

Primary source: [open IET Communications article](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cmu2.12243)

FFSS optimizes fixed-frame slot assignment, while AFSS also adapts frame size. Both use upcoming energy and data, with Poisson energy/data arrivals in the simulation. The publisher reports that both materially improve channel utilization and that AFSS outperforms FFSS because it can resize the frame to the number of qualified nodes.

Comparison verdict: our `FFSS-adapted` baseline is intentionally only an adaptation because the HTA-MAC round abstraction cannot express within-frame position. B12 beats that adaptation by +48.95% FND, +32.12% throughput, and +65.70% packets/J, but loses 18.00% delivery and has 37.84% more stale drops. This is valid only as an internal comparison with the adaptation, not a claim against the published FFSS implementation.

### 5.4 HENO-MAC (Sarang et al., 2024)

Primary full text: [arXiv author manuscript](https://arxiv.org/pdf/2401.00717) and [IEEE DOI](https://doi.org/10.1109/WCNC57260.2024.10571258)

HENO-MAC adapts receiver duty cycle using combined solar-wind energy and an energy-neutral-operation rule.

Key setup differences:

- seven senders and one receiver in a 30 m x 30 m star;
- p-persistent CSMA rather than clustered TDMA slot counts;
- 28-byte packets generated once per second;
- GreenCastalia/TelosB/CC2420, a 3000 mAh receiver battery, and real solar-wind data over 48 hours;
- the paper explicitly accounts for transmit, receive, idle-listen, and sleep states.

Reported results:

- average delay for all packets is reduced by up to **28.5%**;
- highest-priority packet delay is reduced by up to **27.3%**;
- receiver energy reaches 59.8% after two days versus 48.7-50% for the compared single-source protocols.

Comparison verdict: this is strong evidence that hybrid harvesting plus aggressive duty cycling improves delay, but HENO-MAC has no comparable FND, delivery ratio, stale ratio, or packets/J output. HTA-MAC's B12 service deficit means it cannot claim HENO-like delay superiority without adding and reporting packet-delay/AoI metrics.

### 5.5 Eris, Gul, and Boluk (2024): RL slot selection in EH underwater clusters

Primary full text: [Sensors/PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487392/) and [publisher version](https://www.mdpi.com/1424-8220/24/17/5791)

This is the nearest published learned intra-cluster TDMA-slot mechanism, but it is an underwater acoustic network with 100 nodes in a 250 m cube and stochastic piezoelectric harvesting. Nodes choose transmission-slot positions to maximize harvest during idle periods.

Reported results:

- adding harvesting improves FND by **4%**, HND by **14%**, and LND by **22%**;
- harvest-aware RL adds **17%** HND and **38%** LND improvement in the reported comparisons;
- TDMA-EH and TDMA-RL deliver **21%** and **37%** more packets than no-harvesting TDMA;
- RL delivers **11%** more packets than TDMA-EH and captures **96%** of available energy versus **56%** for TDMA-EH.

Comparison verdict: HTA-MAC B12's +62.60% FND and +29.76% throughput versus static are in a strong effect-size range, while its +3.40% FND versus S2A2-adapted is modest. No direct ranking is valid because acoustic propagation, harvest-while-idle semantics, death horizon, and baseline definitions are different.

### 5.6 Farmani et al. (2025): D2PG in clustered EH-WSNs

Primary record: [Springer DOI](https://doi.org/10.1007/s11276-024-03767-5)

D2PG applies deep deterministic policy gradients to continuous transmission-rate control in clustered EH-WSNs.

Reported result: throughput improves by **15.3%**, **12.9%**, and **5.7%** relative to RL, RL-new, and DQN baselines, respectively.

Comparison verdict: B12's +29.76% over static and +22.00% over S2A2-adapted are not proof that HTA-MAC beats D2PG. The actions, baselines, training objective, and evaluation data differ. This paper is evidence that deep RL for clustered EH throughput is established prior art; HTA-MAC's distinction must rest on hybrid HMM transition features, discrete multi-slot allocation, explicit idle energy, hard budgets, and freshness/fairness measurement.

### 5.7 Hasani et al. (2025): DRL throughput control in solar EH-WSNs

Primary full text: [Scientific Reports article](https://www.nature.com/articles/s41598-025-14111-y)

The method uses a shallow DQN to select transmission actions from continuous residual-energy state. Tests include 20-100 nodes, real solar irradiance, 4000-bit packets, and a LEACH-like clustered radio model.

Reported result: throughput improves by **11.79%** over the paper's RL comparison. The authors also report that packet loss increases because the DRL policy injects more traffic, even though more packets reach the BS. Static energy consumption, battery self-consumption, and environmental noise are omitted.

Comparison verdict: the paper exhibits the same failure mode seen in HTA-MAC B12—higher throughput can coexist with worse loss/service behavior. Our explicit stale ratio exposes this trade-off more clearly. HTA-MAC should use that as a design lesson, not treat throughput alone as success.

### 5.8 Sharma, Mastronarde, and Chakareski (2020): structure-aware EH scheduling

Primary manuscript: [arXiv:1807.08315](https://arxiv.org/abs/1807.08315)

The paper learns transmit/defer scheduling for one delay-sensitive EH sensor and exploits monotonic structure in queue backlog and battery state. It closely approximates an offline optimum and improves on conventional Q-learning with lower complexity.

Comparison verdict: no network-level FND or fairness comparison is possible. Its useful implication is architectural: HTA-MAC should test monotonic behavior—more urgent queue state should not systematically receive fewer resources when battery/harvest conditions are otherwise equal.

## 6. Cross-paper performance matrix

| Work | Closest metric | Published gain | Can raw value be compared to HTA-MAC? | Main reason |
|---|---|---:|---|---|
| HTA-MAC B12 | FND vs static | +62.60% | Yes, internal only | identical simulator/schedules |
| HTA-MAC B12 | throughput vs static | +29.76% | Yes, internal only | identical simulator/schedules |
| Ge et al. 2021 | throughput vs tuned static | about +2.35% in stated base case; 16.6-30.1% under battery degradation | No | different nodes, BS, clustering, time base, idle model |
| HENO-MAC 2024 | packet delay | up to -28.5% | No | star CSMA, seven senders, 28-byte packets, 48-hour trace |
| Eris et al. 2024 | FND / throughput | +4% FND; +37% packets vs no-EH TDMA; +11% vs TDMA-EH | No | underwater acoustic and harvest-while-idle model |
| D2PG 2025 | throughput | +5.7-15.3% | No | continuous rate control and different baselines |
| Hasani et al. 2025 | throughput | +11.79% | No | omits static energy; different control and loss definition |
| FFSS/AFSS 2021 | channel utilization | qualitative significant gain | No | within-frame slot position and different metric |
| SHR-TDMA 2020 | waiting-for-slot delay | qualitative significant gain | No | one fixed slot position and analytical arrival model |

The paper values form a context band, not a ranking. HTA-MAC's effect sizes are credible enough to continue, but only identical-simulator baselines can support a superiority statement.

## 7. Where HTA-MAC is genuinely strong

1. **Multi-metric accounting.** The current evaluation exposes throughput, delivery, stale expiry, fairness, idle energy, packets/J, FND, and HND together. Several related papers optimize and report only one or two of these.
2. **Hard resource feasibility.** Every allocation is projected into a cluster slot budget with per-node caps.
3. **Explicit idle-listening cost.** This prevents a free-listening assumption from silently inflating lifetime.
4. **Strong energy-first branch.** B12 is substantially better than static in FND and packets/J and better than the adapted S2A2 policy across the reported vector.
5. **Clear Pareto frontier.** B12 and B16 reveal how additional service capacity exchanges lifetime/efficiency for delivery/freshness.

## 8. Where HTA-MAC is currently weak

1. **B12 is not QoS-feasible.** One joint pass in five is insufficient.
2. **The best balanced result is diagnostic, not trained.** B16 must be retrained from a fresh registered lineage.
3. **Thermal evidence is synthetic auxiliary.** Claims about real hybrid solar-thermal robustness are premature.
4. **No delay/AoI metric is reported.** This prevents a meaningful comparison with HENO-MAC and delay-sensitive EH scheduling papers.
5. **No paper-faithful external baseline is executed.** FFSS and S2A2 are disclosed adaptations, so claims must remain limited to those adaptations.
6. **The current evidence is development-only.** Five schedules are too few for broad claims, and held-out seeds remain locked.

## 9. Recommended next experimental move

### Immediate decision

Proceed with the registered **budget-16 retraining branch**; preserve B12 as the energy-first reference. Do not run held-out seeds yet.

### Required gates before held-out evaluation

1. Train three fresh B16 lineages with the repaired QoS controller.
2. Select using deterministic whole-network evaluation, not training-tail proxies.
3. Require all three absolute constraints—delivery, stale ratio, fairness—to pass, separately from convergence.
4. Compare all baselines at the same budget 16 as well as their native/full-budget settings.
5. Report checkpoints at episodes 400, 450, and 500 to detect late FND regression.
6. Add mean/95th-percentile packet delay or AoI so HENO-MAC and delay-sensitive literature can be discussed on a shared outcome family.
7. Keep categorical boundary-mass, dual-saturation, allocation-equivariance, and constraint-domination audits as mandatory gates.

### Publication-strength follow-up

- Run the once-only held-out evaluation only after B16 passes development gates.
- Report paired seed-level differences and confidence intervals, not only medians.
- Treat censored FND/HND with Kaplan-Meier or common-horizon restricted event-free time.
- Add an exploratory real-trace multi-source test. Until real thermal data are available, label the thermal HMM synthetic auxiliary.
- If resources permit, implement one paper-faithful comparator in a separately preregistered experiment. HENO-MAC is not a good fit for the frozen clustered TDMA simulator; Ge et al.'s clustered solar control or a full within-frame FFSS environment is more defensible.

## 10. Safe and unsafe conclusions

### Safe now

- HTA-MAC B12 provides a strong lifetime/energy-efficiency improvement over identical-simulator baselines, especially static and FFSS-adapted.
- B12 outperforms the S2A2-adapted baseline across all reported development medians.
- B12's delivery and freshness are materially worse than static, energy-proportional, harvest-proportional, and FFSS-adapted policies.
- B16 is the smallest tested diagnostic budget that passes all QoS thresholds on all five development schedules.
- HTA-MAC's internal effect sizes are competitive with the magnitude of gains reported in related EH scheduling literature.

### Unsafe now

- “HTA-MAC beats HENO-MAC, D2PG, Ge et al., or the underwater RL protocol.”
- “HTA-MAC improves network lifetime by more than all published work.”
- “B16 is the final trained model.”
- “The framework is validated for real hybrid solar-thermal harvesting.”
- “Phase 2D QoS passed” or any held-out/statistically significant superiority claim.

## 11. Evidence files

- Trained-result audit: `HTA_MAC_PHASE2D_QOS_TRAINED_RESULTS_ANALYSIS_AND_RECOVERY_PLAN_20260806.md`
- Trained archive: `HTA_MAC_Phase2D_QoS_Trained_Results_20260804.zip`
- Current best checkpoint SHA-256: `70B3682CE3AA1AD1888A4919AC526B603EC8D8CB0C26F2EDEABAA130EAAAD2D7` (seed 4299)
- Development outputs: `outputs/phase3/phase2d_qos_trained_seed4299_dev_eval_20260806/`
- B16 diagnostic: `outputs/phase3/phase2d_qos_trained_seed4299_budget16_dev_diag_20260806/`
- Existing literature archive: `.firecrawl/hta_mac_performance_research_20260804/`

## 12. Final decision

The model is worth continuing. Its current result is not “low performance”; it is a strong but overly conservative energy/lifetime policy. The literature comparison reinforces the same lesson seen in our metrics: throughput or lifetime alone can hide packet loss, delay, or freshness failure. The correct next step is to retrain at B16 with whole-network QoS gates, then perform the locked held-out evaluation once—without weakening thresholds or selecting on held-out results.
