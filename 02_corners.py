"""Step 2 - the corner comparison: classification, windows, models, net benefit.

The exposure here is measured after execution, under either rule. A corner
that was aimed into the box but cut out early is recorded as a short delivery
by the endpoint rule, and its follow-up events differ under the sequence rule
too. Neither rule identifies tactical intent, so nothing below is a causal
estimate; the two rules and the two outcome windows are reported side by side
precisely because they disagree.
"""
import numpy as np
import pandas as pd

from common import (MatchBootstrap, fit_gee, fit_glmm, propensity_match,
                    read_restarts, write_csv, write_json)
from config import BOX_X_THRESHOLDS, BOX_Y, OUTPUT_DIR

OUTCOMES = ["shot", "goal", "strict_shot", "strict_goal"]
COST_OUTCOMES = ["opp_shot30", "opp_goal60"]


def cohens_kappa(a: pd.Series, b: pd.Series) -> tuple[float, float, list]:
    table = pd.crosstab(a, b).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    total = table.to_numpy().sum()
    observed = np.trace(table.to_numpy()) / total
    expected = np.sum(table.sum(axis=1).to_numpy() * table.sum(axis=0).to_numpy()) / total ** 2
    return observed, (observed - expected) / (1 - expected), table.to_numpy().tolist()


def contrasts(frame, boot, label, outcomes):
    rows = []
    treated, control = frame[frame.tr == 1], frame[frame.tr == 0]
    for outcome in outcomes:
        stat = boot.difference(treated, control, outcome)
        rows.append(dict(analysis=label, outcome=outcome, diff_pp=stat["diff"] * 100,
                         lo=stat["lo"] * 100, hi=stat["hi"] * 100,
                         n1=stat["n1"], n0=stat["n0"]))
    return rows


def group_rates(frame, boot, label):
    rows = []
    for value in (1, 0):
        subset = frame[frame.tr == value]
        for outcome in OUTCOMES + COST_OUTCOMES:
            stat = boot.rate(subset, outcome)
            rows.append(dict(analysis=label, group=value, outcome=outcome,
                             n=stat["n"], events=stat["events"],
                             pct=stat["rate"] * 100, lo=stat["lo"] * 100,
                             hi=stat["hi"] * 100))
    return rows


def net_benefit(frame, boot, label):
    """Own 15-s goal rate minus opponent 60-s goal rate, differenced in-resample.

    Reported as a description only. The two sides use different window lengths,
    so they are not on a common scale, and the 1:1 offset is an arbitrary
    weighting rather than the relative match value of a goal conceded.
    """
    treated, control = frame[frame.tr == 1], frame[frame.tr == 0]
    draws = {(col, arm): boot.draws(sub, col)
             for col in ["goal", "opp_goal60"]
             for arm, sub in (("t", treated), ("c", control))}
    net = ((draws[("goal", "t")] - draws[("opp_goal60", "t")])
           - (draws[("goal", "c")] - draws[("opp_goal60", "c")]))
    observed = ((treated.goal.mean() - treated.opp_goal60.mean())
                - (control.goal.mean() - control.opp_goal60.mean()))
    lo, hi = np.quantile(net, [0.025, 0.975])
    return dict(analysis=label, outcome="net_goal", diff_pp=observed * 100,
                lo=lo * 100, hi=hi * 100, n1=len(treated), n0=len(control))


def main() -> None:
    restarts = read_restarts()
    boot = MatchBootstrap(restarts.matchId)
    corners = restarts[restarts.restart == "Corner"].copy()

    # Primary rule: the first same-team event after the corner.
    sequence = corners[corners.sequence != "unclassified"].copy()
    sequence["tr"] = (sequence.sequence == "short").astype(int)
    # Sensitivity rule: whether the pass endpoint fell inside the box.
    coordinate = corners.copy()
    coordinate["tr"] = coordinate.coordinate_short

    contrast_rows, rate_rows, model_rows, psm_summaries = [], [], [], []

    for label, frame in [("sequence", sequence), ("coordinate", coordinate)]:
        print(f"{label} rule: n = {len(frame):,}")
        contrast_rows += contrasts(frame, boot, label, OUTCOMES + COST_OUTCOMES)
        contrast_rows.append(net_benefit(frame, boot, label))
        rate_rows += group_rates(frame, boot, label)

        for outcome in OUTCOMES:
            model_rows.append(fit_gee(frame, outcome, label))
            print(f"  GEE {outcome}: OR = {model_rows[-1]['estimate']:.3f}")
        for outcome in ["shot", "goal"]:
            model_rows.append(fit_glmm(frame, outcome, label))
            print(f"  GLMM {outcome}: OR = {model_rows[-1]['estimate']:.3f}"
                  f"  sigma = {model_rows[-1]['sigma']:.4f}")

        matched, balance, support, summary, scored = propensity_match(frame, label)
        contrast_rows += contrasts(matched, boot, f"{label}_PSM", OUTCOMES)
        psm_summaries.append(summary)
        write_csv(balance, f"psm_balance_{label}.csv")
        write_csv(support, f"psm_support_{label}.csv")
        write_csv(scored[["eventId", "matchId", "teamId", "competition", "tr",
                          "ps", "logit_ps", "matched"]], f"psm_scores_{label}.csv")
        print(f"  PSM: {summary['pairs']:,} pairs, max |SMD| = {summary['max_abs_smd']:.3f}")

    # Endpoint-threshold sensitivity.
    for threshold in BOX_X_THRESHOLDS:
        frame = corners.copy()
        frame["tr"] = (~((frame.end_x >= threshold)
                         & frame.end_y.between(*BOX_Y))).astype(int)
        contrast_rows += contrasts(frame, boot, f"threshold_{threshold}", OUTCOMES)

    # Classification agreement. Kappa measures how far two operational rules
    # agree; without video ground truth it cannot establish that either is valid.
    observed, kappa, table = cohens_kappa(sequence.coordinate_short, sequence.tr)
    unclassified = corners[corners.sequence == "unclassified"]
    write_json(dict(n=len(sequence), table=table, agreement=observed, kappa=kappa,
                    unclassified_n=len(unclassified),
                    unclassified_goal_pct=float(unclassified.goal.mean() * 100),
                    unclassified_share=float(len(unclassified) / len(corners))),
               "corner_classification.json")
    print(f"agreement {observed * 100:.2f}%  kappa {kappa:.3f}  "
          f"unclassified {len(unclassified):,} (goal rate "
          f"{unclassified.goal.mean() * 100:.2f}%)")

    # Sensitivity: drop matches whose reconstructed score never reconciled.
    quality = pd.read_csv(OUTPUT_DIR / "score_reconciliation.csv")
    unreconciled = set(quality.loc[~quality.corrected_match, "matchId"])
    for label, frame in [("sequence", sequence), ("coordinate", coordinate)]:
        subset = frame[~frame.matchId.isin(unreconciled)]
        for outcome in ["goal", "strict_goal"]:
            model_rows.append(fit_gee(subset, outcome, f"{label}_reconciled_only"))

    write_csv(pd.DataFrame(contrast_rows), "corner_contrasts.csv")
    write_csv(pd.DataFrame(rate_rows), "corner_rates.csv")
    write_csv(pd.DataFrame(model_rows), "corner_models.csv")
    write_json(psm_summaries, "psm_summary.json")


if __name__ == "__main__":
    main()
