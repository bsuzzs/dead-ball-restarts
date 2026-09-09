# Set-piece restart analysis

Reproducible computation for 193,197 dead-ball restarts from 1,941 elite football matches.
This repository contains research code, aggregate reference results, and validation tools.
Raw provider data, local environments, and manuscript files are not included.

## Run the complete study

Use Python 3.14 (validated on 3.14.7). From this repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export RESTART_DATA="/path/to/raw-data"
python verify_inputs.py
python run_all.py
python verify_reproduction.py
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and set
`$env:RESTART_DATA = "C:\path\to\raw-data"` instead of the shell commands above.

The exact data layout is in [data/README.md](data/README.md). Input verification
requires the 520 files recorded in `data/source_sha256.json`; it does not download data.
Set `RESTART_OUT` to change the output directory (default: `outputs/`). Both pipeline
and result verifier use this setting. Raw input files are read only.

The full study requires SHAP and PyTorch. Some original steps skip diagnostics when
these are missing; `verify_reproduction.py` fails if their expected outputs are missing,
so a skipped diagnostic cannot count as a complete reproduction.

To run selected steps after their upstream outputs exist: `python run_all.py 02 03`.

## Pipeline

| Step | Analysis |
| --- | --- |
| 00 | Event extraction, outcome windows, goal attribution and score reconciliation |
| 01 | Seven-category benchmarks, window sensitivity, spatial tables |
| 02 | Corner comparisons, GEE, GLMM, bootstrap and propensity matching |
| 03 | Throw-in zone, direction and displacement comparisons |
| 04 | Match outcomes and corrected/uncorrected clustered Wald test |
| 05 | IMPECT, StatsBomb and Understat external comparisons |
| 06 | Match-grouped classification and SHAP |
| 07 | Post-corner GRU diagnostic |
| 08 | Figures 1–7 and S1–S5: PNG, PDF, SVG, TIFF |
| 09 | Selected results and internal consistency checks |

See [output definitions](docs/outputs.md) and [validation notes](docs/VALIDATION.md).

## Outcome and xG definitions

Binary outcomes use the first same-team shot within 15 seconds in the same period.
The strict variant stops at the first opposing-team event. Direct free-kick shots and
penalties are themselves shots. Corner classifications are measured after execution;
the resulting comparisons are not causal effects of tactical intention. The GRU uses
post-corner events and is a diagnostic, not evidence of pre-restart predictive validity.

For StatsBomb xG, pass restarts sum same-team shot xG within 15 seconds in the same
period. Direct free-kick shots and penalties count only the restart shot itself.
Each quantity is averaged over all restarts of that type; no shot contributes zero.
The code also provides first-shot and shot-conditional alternatives. The `window`
column names retain this explicit self-shot exception. No analysis values were changed
when clarifying this definition.

Corner region statistics cover all 19,316 corners with endpoints; Figure 3 displays
18,278 with endpoint x ≥ 60. The denominator JSON records both. Window tables retain
full precision; round only at display. The manuscript uses the finite-sample-corrected
clustered Wald statistic.

## Validation status

The supplied analysis was run end to end against the recorded local data on
2026-09-09. Six AUCs reproduced the manuscript: 0.849, 0.852, 0.861, 0.859,
0.883 (GRU), and 0.648 (corner tabular baseline). Fresh installation on another machine
and provider-data retrieval have not been tested. Numerical libraries and hardware
can cause differences; verification tolerances and failures are explicit.

Reference tables are frozen outputs from that verified run. The verifier compares
new output to them and does not copy them into the analysis. It does not certify every
sentence or number in a manuscript. Raw data must be obtained separately.

## Citation and reuse

See [CITATION.md](CITATION.md). No author identities, journal acceptance, DOI or
repository URL have been invented. No software reuse licence is assigned in this
package; the authors can add their chosen licence independently of provider data terms.
