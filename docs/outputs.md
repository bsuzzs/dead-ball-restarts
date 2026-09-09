# Output files

Everything below is written to `outputs/` and is git-ignored. Sizes are
approximate for the published sample.

## From `00_extract.py`

| File | Contents |
| --- | --- |
| `restarts.parquet` (or `restarts.csv.gz`) | One row per restart: identifiers, coordinates, both outcome windows, opponent counter outcomes, classification labels, covariates |
| `team_matches.csv` | One row per team-match: result, venue, pre-match points per game for both sides |
| `score_reconciliation.csv` | Per match: official score, naive tag-101 score, corrected score, and whether each reconciles |
| `goal_attribution.csv` | Per team-match: goals split into restart, open play, penalty, own goal |
| `source_manifest.json` | SHA-256 of every raw event file, event counts, free-kick sub-type counts |
| `software_versions.json` | Package versions actually present at run time |

## From `01_benchmarks.py`

`benchmarks.csv` (Table 2), `window_sensitivity.csv`, `corner_zones.csv`,
`freekick_distance_bands.csv`, `throwin_x_bands.csv`, `throwin_x_fine.csv`,
`corner_by_competition.csv`.

## From `02_corners.py`

`corner_contrasts.csv` (rate differences and the net-benefit summary),
`corner_rates.csv` (group rates for both rules and both windows),
`corner_models.csv` (GEE and GLMM odds ratios, plus reconciled-only
sensitivity), `corner_classification.json` (agreement, κ, unclassified group),
`psm_balance_*.csv`, `psm_support_*.csv`, `psm_scores_*.csv`,
`psm_summary.json`.

## From `03_throwins.py`

`throwin_contrasts.csv`, `throwin_models.csv`, `throwin_length_bands.csv`,
`attacking_third_throwins.csv`.

## From `04_matches.py`

`match_results.csv` (win/draw/loss by goal-source advantage, both restart and
open-play comparators), `single_goal_test.json`.

## From `05_external.py`

`impect_rates.csv`, `impect_restarts.csv`, `statsbomb_corner_rates.csv`,
`statsbomb_xg_by_restart.csv`, `statsbomb_penalties.csv`,
`statsbomb_restarts.csv`, `understat_shots.csv`.

## From `06_predictive.py` / `07_sequence_model.py`

`predictive_checks.json`, `shap_importance_shot.csv`,
`shap_importance_goal.csv`, `sequence_model.json`.

## From `08_figures.py`

`figures/Figure1..7` and `figures/FigureS1..S5`, each in PNG, PDF, SVG and
TIFF; `corner_map_bins.json` records the spread of hexagonal bin rates, which
is what keeps the text from quoting a peak bin as if it were a zone rate.

## From `09_audit.py`

`manuscript_numbers.json` — selected study results and internal consistency checks. This is not a full
manuscript audit. Use verify_reproduction.py for reference-result comparison.

## Two conventions the tables must state explicitly

**Corner denominators.** `corner_zones.csv` covers every corner with a
recorded endpoint (19,316). The landing map in Figure 3 restricts to endpoints
with `x >= 60` (18,278). `corner_denominators.json` records both, so a table
caption can be generated rather than typed; quoting the map's n for the zone
table is the mistake this file exists to prevent.

**StatsBomb xG.** The four output fields preserve the manuscript computation:

| Column | Definition |
| --- | --- |
| `mean_window_xg_per_restart` | Pass restarts: sum same-team shot xG within 15 s in the same period; direct free-kick shots/penalties: restart shot xG only. Average across all restarts of that type; no shot contributes zero. |
| `mean_first_shot_xg_per_restart` | First-shot xG, averaged across all restarts; self-shot restarts use the restart shot. |
| `mean_window_xg_given_shot` | The first quantity conditional on a restart producing a shot. The self-shot exception still applies. |
| `mean_first_shot_xg_given_shot` | First-shot xG conditional on a restart producing a shot; not an average across every shot in the window. |

The legacy word `window` in the column names does not override the self-shot exception.
These are not a uniform all-shot window measure across all restart categories.

**Cluster-robust Wald test.** `single_goal_test.json` gives corrected and uncorrected
results, the factor `c = [G/(G-1)] * [(N-1)/(N-K)]`, and the cluster count.
The manuscript uses the corrected statistic.
