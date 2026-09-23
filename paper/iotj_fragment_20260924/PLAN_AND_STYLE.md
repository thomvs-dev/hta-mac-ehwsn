# IEEE Internet of Things Journal manuscript plan

## Reference-paper analysis

The supplied HEART-CH manuscript is nine US-letter pages in an IEEE journal
two-column format. Its progression is application motivation, system model,
framework, evaluation, and conclusion. It defines notation before algorithms,
places assumptions beside equations, uses Roman-numbered sections and lettered
subsections, and discusses each figure through a concrete performance trade-off.
The abstract moves from the energy problem to architecture and measured results.
It separates clustering from routing and reports baseline strengths as well as
the proposed method's strengths. Those structural choices are useful here.

Use original wording and newly generated diagrams. Do not copy the HMM/Rainbow
architecture, plots, experimental values, or mobile-topology claims into the
fragment study. The supplied document has no visible author block or publication
identifier; cite it as an unpublished manuscript, not an IEEE publication.

## Selected scope and argument

Title: Energy-Feasible Fragment Recovery for Clustered IoT Networks:
Fixed-Window Accounting and Shared-Head Evaluation.

The contribution is an explicitly accounted, reproducible comparative MAC study:
local bitmap information, jointly funded fixed windows, finite source/relay
buffers, receipt-based custody, and shared-head integration. Selective recovery
and block acknowledgements are established mechanisms. Explain what the evaluated
design adds to the experimental contract without claiming their invention.

Sections: introduction; related work; system model; fragment-recovery method;
properties and complexity; experimental protocol; results/discussion; conclusion.
Use IoT terminology throughout original prose, retaining exact published titles
in the bibliography. Do not call the deterministic method reinforcement learning.
No training-status sentence is needed for a method described accurately as
deterministic. Include material assumptions and negative variants in the results.

Evidence: fixed (24576 episodes), guard (30720), shared (9216), totaling64512
valid episodes and160 distinct paired development seeds. Preserve the invalidated
predecessor exclusion. All figures are derived from shipped raw evidence; any
new subgroup summaries are descriptive and do not change the frozen gates.

Deliver: an8--10page compiled IEEEtran journal PDF, editable LaTeX, vector figures,
machine-readable derived plot data, source-verification notes and a review memo.
No changes to frozen V1 or published experimental sources/configuration/results.
Author block awaits names/affiliations. No automatic submission or new Git push.

## Verification

Recheck public raw-evidence hashes and statistics. Check equations against the
executable model, including fixed padding, total head buffer cap and absorbing
death. Compile until citations resolve. Inspect all rendered pages, page count,
overfull boxes, tables, figure labels and reference numbering. Keep journal
readiness concerns in an author-facing memo as well as necessary scientific
scope qualifications in the manuscript.
