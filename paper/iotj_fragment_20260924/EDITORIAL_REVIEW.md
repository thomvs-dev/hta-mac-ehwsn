# Author review before IEEE Internet of Things Journal submission

The manuscript is a polished, evidence-based draft, not an assurance of journal
acceptance. Its supported contribution is the execution/accounting framework
and matched comparative evaluation of known fragment-recovery mechanisms. The
main scientific review risk is whether that contribution is sufficiently
substantial for IoT Journal; wording alone cannot resolve this.

## Required author information

Supply names, affiliations, corresponding-author email and ORCIDs. Verify the
HEART-CH manuscript's authors, status, title and whether it has a citable public
identifier. The present reference is explicitly unpublished and does not invent
an IEEE venue, DOI, or authorship. Review all methods and calculations personally.

## Highest-value scientific extensions

The statistical revision confirms consistent shared-head gains across all32
seeds and supplies global Holm-adjusted paired tests and exact sign sensitivity.
See `STATISTICAL_REVIEW.md`. Strong within-model significance does not remove
the novelty, matched-baseline, or external-validity concerns below. TableI now
compares reported operating points from three recent papers with evidence types
explicitly distinguished; no cross-paper superiority is inferred.

1. Integrate the frozen fragment baseline into actual exogenous HEART-CH traces,
   preserving wire time, queues, energy costs, and custody semantics. The current
   dedicated-head model is not that integration.
2. Add a matched implementation of a relevant published mechanism or standard,
   then freeze a new untouched confirmation cohort. Do not compare external
   papers' percentages as if their simulators were identical.
3. Examine the shared-head resource bottleneck with preregistered interventions:
   separate head energy from payload-buffer capacity and distinguish head-switch
   disruption from access to a less-used battery. The present subgroup patterns
   are descriptive, not causal attribution.
4. Validate control losses, longer horizons, measured harvest traces and radio
   costs. No hardware is available in the current project. Simulation claims
   should remain identified as simulation.

The existing failed prefix and guard variants remain in the manuscript, and the
timing-invalidated predecessor remains excluded. Original Gate A/B requirements
are not redefined or certified by the component study. The paper accurately
describes a deterministic mechanism and therefore needs no irrelevant narrative
about neural-training status.

## Formatting and submission details

The official journal guidelines require double-column IEEE format and a150--250
word single-paragraph abstract. They state mandatory charges after eight
published pages; check the current policy before submission. Author identities
are required for the journal's single-anonymous review process.

Source: https://ieee-iotj.org/guidelines-for-authors/ (checked24September2026).

The draft includes a concise acknowledgment of Codex assistance in drafting,
LaTeX, and figure-generation code, consistent with IEEE's AI-content disclosure
guidance. The authors should verify the full applicable disclosure before filing.
Source: https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/

## Citation verification

- Iannello et al.: author preprint arXiv1112.2409 and its linked journal DOI.
- RFC4944, RFC8930, RFC8931, RFC7554 and RFC8180: RFC Editor originals.
- RI-MAC: Rice University author PDF and publisher-linked bibliographic record.
- Scanzio et al.: arXiv2411.12879; cited as a preprint, not inferred publication.
- Van Leemput et al.: accepted author manuscript, institutional metadata and
  author repository; IEEE Access12,180034--180047,2024.
- RL-ASL: arXiv2604.07533v2; cited as a preprint, without inventing journal pages.
- CC2420: Texas Instruments product specifications.
- Efron: Annals of Statistics7(1),1--26,1979, DOI10.1214/aos/1176344552.
- HEART-CH: the supplied nine-page PDF; unpublished status awaits author details.
- Experiment archive: immutable public Git commit0e12c49, verified locally.

Firecrawl Research failed once with ECONNREFUSED127.0.0.1:9. Verification used
primary publisher, RFC, manufacturer, author, institutional and arXiv sources.
No assertion is made that every relevant paper was reviewed.
