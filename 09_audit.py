"""Step 9: collect selected study results and run internal consistency checks.

This is not a comparison against every number in the manuscript.
Use verify_reproduction.py for comparison with the bundled reference results.
"""
import json

import numpy as np
import pandas as pd

from common import read_restarts, write_json
from config import EXPECTED_MATCHES, EXPECTED_RESTARTS, EXPECTED_TEAMS, OUTPUT_DIR


def load(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    return (json.loads(path.read_text()) if path.suffix == ".json"
            else pd.read_csv(path))


def pick(frame, **conditions):
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= frame[column].to_numpy() == value
    rows = frame[mask]
    if len(rows) != 1:
        raise KeyError(f"expected one row for {conditions}, got {len(rows)}")
    return rows.iloc[0]


def ci(row, value="diff_pp", digits=2):
    return f"{row[value]:.{digits}f} [{row['lo']:.{digits}f}, {row['hi']:.{digits}f}]"


def main() -> None:
    restarts = read_restarts()
    benchmarks = load("benchmarks.csv")
    contrasts = load("corner_contrasts.csv")
    models = load("corner_models.csv")
    rates = load("corner_rates.csv")
    classification = load("corner_classification.json")
    psm = {d["analysis"]: d for d in load("psm_summary.json")}
    quality = load("score_reconciliation.csv")
    throwin = load("throwin_contrasts.csv")
    throwin_models = load("throwin_models.csv")
    results = load("match_results.csv")
    single = load("single_goal_test.json")
    windows = load("window_sensitivity.csv")
    zones = load("corner_zones.csv")
    understat = load("understat_shots.csv")

    # Internal consistency conditions.
    checks = {
        "restart_count": len(restarts) == EXPECTED_RESTARTS,
        "match_count": restarts.matchId.nunique() == EXPECTED_MATCHES,
        "team_count": restarts.teamId.nunique() == EXPECTED_TEAMS,
        "event_ids_unique": bool(restarts.eventId.is_unique),
        "strict_within_loose": bool((restarts.strict_shot <= restarts.shot).all()
                                    and (restarts.strict_goal <= restarts.goal).all()),
        "goals_within_shots": bool((restarts.goal <= restarts.shot).all()),
        "benchmark_n_sums_to_total": int(benchmarks.n.sum()) == len(restarts),
        "psm_pairs_unique": all(d["pairs"] > 0 for d in psm.values()),
    }
    if not all(checks.values()):
        raise AssertionError(f"consistency checks failed: "
                             f"{[k for k, v in checks.items() if not v]}")

    corner = pick(benchmarks, restart="Corner")
    corner_window = pick(windows, restart="Corner")
    seq_short = pick(rates, analysis="sequence", group=1, outcome="goal")
    seq_other = pick(rates, analysis="sequence", group=0, outcome="goal")

    numbers = {
        "sample": {
            "restarts": len(restarts),
            "matches": int(restarts.matchId.nunique()),
            "teams": int(restarts.teamId.nunique()),
            "by_category": benchmarks.set_index("label").n.to_dict(),
        },
        "table2_percent": {
            row.label: {
                "n": int(row.n),
                "shot_15s": round(row.shot_pct, 2), "goal_15s": round(row.goal_pct, 2),
                "shot_strict": round(row.strict_shot_pct, 2),
                "goal_strict": round(row.strict_goal_pct, 2),
            } for row in benchmarks.itertuples()
        },
        "window_sensitivity_corner": {
            "shot": [round(corner_window[f"shot{w}"], 2) for w in (10, 15, 20)],
            "goal": [round(corner_window[f"goal{w}"], 2) for w in (10, 15, 20)],
        },
        "corner_classification": {
            "sequence_n": classification["n"],
            "short_n": int(pick(rates, analysis="sequence", group=1, outcome="goal").n),
            "other_n": int(pick(rates, analysis="sequence", group=0, outcome="goal").n),
            "agreement_pct": round(classification["agreement"] * 100, 2),
            "kappa": round(classification["kappa"], 3),
            "unclassified_n": classification["unclassified_n"],
            "unclassified_goal_pct": round(classification["unclassified_goal_pct"], 2),
        },
        "headline_corner_comparison": {
            "short_goal_pct_15s": round(seq_short.pct, 2),
            "other_goal_pct_15s": round(seq_other.pct, 2),
            "rate_difference_15s_pp": ci(pick(contrasts, analysis="sequence", outcome="goal")),
            "rate_difference_strict_pp": ci(pick(contrasts, analysis="sequence",
                                                 outcome="strict_goal")),
            "gee_or_15s": ci(pick(models, analysis="sequence", model="GEE", outcome="goal"),
                             "estimate"),
            "gee_or_strict": ci(pick(models, analysis="sequence", model="GEE",
                                     outcome="strict_goal"), "estimate"),
            "glmm_or_15s": ci(pick(models, analysis="sequence", model="GLMM_GH",
                                   outcome="goal"), "estimate"),
            "matched_difference_pp": ci(pick(contrasts, analysis="sequence_PSM",
                                             outcome="goal")),
            "net_goal_pp": ci(pick(contrasts, analysis="sequence", outcome="net_goal")),
        },
        "endpoint_rule_comparison": {
            "rate_difference_15s_pp": ci(pick(contrasts, analysis="coordinate", outcome="goal")),
            "rate_difference_strict_pp": ci(pick(contrasts, analysis="coordinate",
                                                 outcome="strict_goal")),
            "gee_or_15s": ci(pick(models, analysis="coordinate", model="GEE",
                                  outcome="goal"), "estimate"),
            "net_goal_pp": ci(pick(contrasts, analysis="coordinate", outcome="net_goal")),
        },
        "propensity_matching": {
            rule: {"pairs": d["pairs"], "max_abs_smd": round(d["max_abs_smd"], 3)}
            for rule, d in psm.items()
        },
        "corner_landing_zones_pct": {
            row.zone: round(row.goal_pct, 2) for row in zones.itertuples()
        },
        "corner_denominators": load("corner_denominators.json"),
        "throw_ins": {
            "attacking_third_shot_pct": round(pick(throwin, analysis="zone",
                                                   outcome="shot").pct_1, 2),
            "elsewhere_shot_pct": round(pick(throwin, analysis="zone",
                                             outcome="shot").pct_0, 2),
            "zone_gee_shot_or": ci(pick(throwin_models, analysis="zone", outcome="shot"),
                                   "estimate"),
            "forward_shot_difference_pp": ci(pick(throwin, analysis="attacking_third_forward",
                                                  outcome="shot")),
            "forward_gee_shot_or": ci(pick(throwin_models, analysis="attacking_third_forward",
                                           outcome="shot"), "estimate"),
            "long_shot_difference_pp": ci(pick(throwin, analysis="attacking_third_long",
                                               outcome="shot")),
            "long_gee_shot_or": ci(pick(throwin_models, analysis="attacking_third_long",
                                        outcome="shot"), "estimate"),
        },
        "match_results": {
            "reconciled_matches": int(quality.corrected_match.sum()),
            "naive_reconciled_matches": int(quality.raw_tag101_match.sum()),
            "total_matches": len(quality),
            "restart_more_win_pct": round(pick(results, source="setpiece",
                                               comparison="more").win_pct, 2),
            "openplay_more_win_pct": round(pick(results, source="openplay",
                                                comparison="more").win_pct, 2),
            "single_restart_wdl": [int(pick(results, source="single_setpiece",
                                            comparison="only_goal")[c])
                                   for c in ("wins", "draws", "losses")],
            "single_openplay_wdl": [int(pick(results, source="single_openplay",
                                             comparison="only_goal")[c])
                                    for c in ("wins", "draws", "losses")],
            "chi2": round(single["chi2"], 3), "p": round(single["p"], 3),
            "cluster_wald": round(single["cluster_wald"], 3),
            "cluster_p": round(single["cluster_p"], 3),
            "cluster_wald_uncorrected": round(single.get("cluster_wald_uncorrected", float("nan")), 3),
            "cluster_convention": single.get("cluster_convention"),
        },
        "consistency_checks": checks,
    }

    if understat is not None:
        corners_only = understat[understat.situation == "FromCorner"].set_index("season")
        numbers["understat_corners"] = {
            int(season): {"shots": int(row.shots), "goals": int(row.goals),
                          "goals_per_match": round(row.goals_per_match, 3),
                          "conversion_pct": round(row.conversion_pct, 2),
                          "xg_per_shot": round(row.xg_per_shot, 4)}
            for season, row in corners_only.iterrows()
        }

    for name, extra in [("impect_rates.csv", "impect"),
                        ("statsbomb_corner_rates.csv", "statsbomb"),
                        ("sequence_model.json", "sequence_model"),
                        ("predictive_checks.json", "predictive_checks")]:
        payload = load(name)
        if payload is None:
            continue
        numbers[extra] = (payload if isinstance(payload, dict)
                          else payload.round(3).to_dict("records"))

    write_json(numbers, "manuscript_numbers.json")
    print(json.dumps(numbers["headline_corner_comparison"], indent=2))
    print("\nall consistency checks passed")


if __name__ == "__main__":
    main()
