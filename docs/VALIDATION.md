# Validation record

Validated on 2026-09-09 using the versions in `tested_environment.json`.

The submitted analysis was executed from an empty output directory through steps
00–09. It completed successfully; all 520 raw inputs and the submitted source files
were unchanged. Six model AUCs matched the manuscript's reported precision.

The GitHub preparation preserves computation: syntax-tree comparison excluding
comments/docstrings confirmed that every original Python file except the plotting
file has unchanged executable content. In the plotting file, Figure 3's axes and
colour bar were moved inward to prevent right-edge clipping; bins, colour scale,
data and aggregation were not changed. The corrected Figure 6 layout is retained.
Final figures and result summaries were regenerated, then compared with references.

`verify_inputs.py` checks the SHA-256 of all 520 required raw source files.
`verify_reproduction.py` checks 32 aggregate result files and requires all 48 figure
files to exist. Numeric tolerance: relative 1e-6, absolute 1e-8; CSV row/column order
must match. A mismatch or missing file gives a nonzero exit code. Figure presence
checks are not pixel comparisons or visual certification. The 32 references include
all six AUCs, SHAP rankings, benchmark rates, corner/throw-in models, matching
summaries, external summaries, and the corrected and uncorrected Wald statistics.

References are aggregate research results; they do not include restart/event tables,
individual propensity scores, team-match tables, manuscript files or raw provider data.
The input hash manifest identifies source files but does not contain their contents.

An independent audit of the supplied manuscript checked 613 numeric table cells:
610 matched, three Table S1 display-rounding differences remained in the supplied
DOCX. The manuscript updates are listed in `MANUSCRIPT_NOTES_ZH.md`. This package
therefore claims reproducible computation, not certification of an unchanged manuscript.

The existing environment was used; fresh dependency installation and independent
reacquisition of provider data were not tested. The recorded exact dependency versions
are the tested configuration, not a guarantee of availability on all platforms.
