# IEEE IoT Journal manuscript draft

Target: IEEE Internet of Things Journal. Standard IEEEtran journal class,
10-point, US-letter, two-column format. The abstract is206 words. No font or
margin compression is used to meet the requested8--10page length.

`main.tex` is the editable manuscript; references are embedded in IEEE numeric
order. `figures/` contains five original vector PDF figures generated from the
published raw evidence. The sixth numbered figure contains the algorithm.
`derived_results.json` contains the exploratory marginal summaries used by the
resource-sensitivity plot. These summaries do not change any experimental gate.
`PLAN_AND_STYLE.md` records the prior analysis of the supplied HEART-CH paper.
`EDITORIAL_REVIEW.md` lists author actions before submission.

The statistical revision adds `STATISTICAL_REVIEW.md`, all 22 paired comparisons
in `statistical_analysis.json` and `statistical_tests.csv`, and all per-seed
endpoints in `seed_endpoints.csv`. Reproduce with `analyze_statistics.py`, then
`write_statistical_report.py` and `verify_statistics.py` (NumPy/SciPy required).
These are post hoc supplementary tests with global Holm correction, not new
confirmation or changes to the original gates. The original ratio-of-sums
bootstrap analysis remains distinct from these mean seed-ratio tests.

Build from this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

MiKTeX may require access to its local cache. The local build used
`--disable-installer` to avoid a stalled automatic package installation. The
document uses only IEEEtran, amsmath, amssymb, graphicx, booktabs, array, cite,
and url. No bibliography program is required. On Overleaf, upload main.tex and
the figures directory, select pdfLaTeX, and compile twice.

To regenerate plots in the full research repository:

```powershell
python -B paper/iotj_fragment_20260924/build_figures.py
python -B tools/verify_fragment_publication_20260921.py
```

Python3.13 and NumPy2.4.4 were used for the original evidence, with Matplotlib
for these plots. The reference PDF was inspected for organization and layout;
its prose, numerical results, HMM/Rainbow method and graphics were not copied.

Original prose uses IoT terminology. Published reference titles retain their
correct wording. HEART-CH is provisionally cited as an unpublished supplied
manuscript because no authors or publication metadata were provided. The blank
author block is intentional and must be filled before submission.

The revised delivery is `output/pdf/iotj_fragment_recovery_20260924_statistics.pdf`;
the earlier PDF remains available unchanged.
The companion source ZIP contains the manuscript, vector figures, derivation
data, figure script, and review notes, excluding intermediate compiler files.
No frozen V1 artifact or published experiment file was altered. The build scripts
do not submit the manuscript. The next-study plan is in
`reports/FRAGMENT_NEXT_STUDY_PLAN_20260924.md` in the repository.
