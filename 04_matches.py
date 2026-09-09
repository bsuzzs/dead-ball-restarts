"""Step 4 - goal source and match outcome.

"Teams that scored more restart goals than their opponent won 68.8% of the
time" is close to a tautology: conditioning on a goal advantage of any kind
predicts winning. The comparison that actually discriminates is the
single-goal test, so an open-play comparator is computed on the same sample.
"""
import numpy as np
import pandas as pd

from common import write_csv, write_json
from config import OUTPUT_DIR

SOURCES = ["setpiece", "openplay"]


def multinomial_cluster_test(frame):
    """Wald test on the set-piece indicator from a multinomial logit,
    with a cluster-robust (match-level) covariance.

    The two teams in a match appear as separate rows and their results are
    mechanically dependent, so match clustering is needed alongside the
    exact-count chi-squared test.
    """
    import statsmodels.api as sm
    from scipy.stats import chi2

    outcome = frame.result.map({"W": 0, "D": 1, "L": 2}).to_numpy()
    design = sm.add_constant(frame.is_setpiece.to_numpy(float), has_constant="add")
    model = sm.MNLogit(outcome, design)
    fitted = model.fit(disp=0)

    n_exog = design.shape[1]
    n_eq = np.asarray(fitted.params).shape[1]          # J - 1 equations
    bread = np.asarray(fitted.cov_params())
    if bread.shape != (n_exog * n_eq, n_exog * n_eq):
        raise RuntimeError(f"unexpected covariance shape {bread.shape}")

    # statsmodels stacks MNLogit parameters equation by equation, so the
    # indicator (exog column 1) sits at position j * n_exog + 1 in equation j.
    idx = [j * n_exog + 1 for j in range(n_eq)]

    scores = np.asarray(model.score_obs(np.asarray(fitted.params)))
    if scores.shape != (len(frame), n_exog * n_eq):
        raise RuntimeError(f"unexpected score shape {scores.shape}")
    meat = np.zeros_like(bread)
    n_clusters = 0
    for _, rows in pd.DataFrame(scores).groupby(frame.matchId.to_numpy()):
        total = rows.to_numpy().sum(axis=0)
        meat += np.outer(total, total)
        n_clusters += 1

    # Finite-sample correction. Conventions differ between packages and the
    # choice moves the statistic by a fraction of a percent, so it is applied
    # explicitly and both versions are returned rather than left implicit.
    # c = [G/(G-1)] * [(N-1)/(N-K)] is the Stata `vce(cluster)` convention.
    n_obs, n_params = len(frame), bread.shape[0]
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))

    beta = np.asarray(fitted.params).ravel(order="F")[idx]
    out = {}
    for name, factor in [("corrected", correction), ("uncorrected", 1.0)]:
        vcov = (bread @ (meat * factor) @ bread)[np.ix_(idx, idx)]
        stat = float(beta @ np.linalg.solve(vcov, beta))
        out[name] = (stat, float(chi2.sf(stat, len(beta))))
    out["clusters"] = n_clusters
    out["correction_factor"] = float(correction)
    return out


def main() -> None:
    quality = pd.read_csv(OUTPUT_DIR / "score_reconciliation.csv")
    reconciled = set(quality.loc[quality.corrected_match, "matchId"])
    teams = pd.read_csv(OUTPUT_DIR / "team_matches.csv")
    attribution = pd.read_csv(OUTPUT_DIR / "goal_attribution.csv")
    frame = teams.merge(attribution, on=["matchId", "teamId"], validate="one_to_one")
    frame = frame[frame.matchId.isin(reconciled)].copy()
    print(f"reconciled matches: {frame.matchId.nunique():,}")

    lookup = frame.set_index(["matchId", "teamId"])
    rows = []
    for source in SOURCES:
        opponent = np.array([lookup.loc[(r.matchId, r.oppId), source]
                             for r in frame.itertuples()])
        sign = np.sign(frame[source].to_numpy() - opponent)
        for value, label in [(1, "more"), (0, "equal"), (-1, "fewer")]:
            subset = frame[sign == value]
            counts = subset.result.value_counts()
            rows.append(dict(source=source, comparison=label, n=len(subset),
                             wins=int(counts.get("W", 0)), draws=int(counts.get("D", 0)),
                             losses=int(counts.get("L", 0)),
                             win_pct=float(counts.get("W", 0) / max(len(subset), 1) * 100)))

    # Single-goal test: exactly one goal, from one source, no penalty or own goal.
    only_setpiece = frame[(frame.setpiece == 1) & (frame.openplay == 0)
                          & (frame.penalty == 0) & (frame.owngoal == 0)]
    only_openplay = frame[(frame.setpiece == 0) & (frame.openplay == 1)
                          & (frame.penalty == 0) & (frame.owngoal == 0)]
    table = []
    for name, subset in [("single_setpiece", only_setpiece), ("single_openplay", only_openplay)]:
        counts = subset.result.value_counts()
        row = [int(counts.get(k, 0)) for k in ["W", "D", "L"]]
        table.append(row)
        rows.append(dict(source=name, comparison="only_goal", n=len(subset),
                         wins=row[0], draws=row[1], losses=row[2],
                         win_pct=float(row[0] / max(len(subset), 1) * 100)))

    from scipy.stats import chi2_contingency
    chi2_stat, p_value, dof, expected = chi2_contingency(table)

    combined = pd.concat([only_setpiece.assign(is_setpiece=1),
                          only_openplay.assign(is_setpiece=0)])
    wald_result = multinomial_cluster_test(combined)
    wald, wald_p = wald_result["corrected"]

    write_csv(pd.DataFrame(rows), "match_results.csv")
    write_json(dict(table=table, chi2=float(chi2_stat), p=float(p_value), df=int(dof),
                    expected=expected.tolist(), n=len(combined),
                    matches=int(combined.matchId.nunique()),
                    cluster_wald=wald, cluster_p=wald_p,
                    cluster_wald_uncorrected=wald_result["uncorrected"][0],
                    cluster_p_uncorrected=wald_result["uncorrected"][1],
                    cluster_correction_factor=wald_result["correction_factor"],
                    clusters=wald_result["clusters"],
                    cluster_convention="c = [G/(G-1)] * [(N-1)/(N-K)]",
                    note="Failing to reject equality is not evidence of equivalence."),
               "single_goal_test.json")
    print(f"single-goal test: chi2 = {chi2_stat:.3f}, p = {p_value:.3f}")
    print(f"clustered Wald (corrected) = {wald:.3f}, p = {wald_p:.3f}; "
          f"uncorrected = {wald_result['uncorrected'][0]:.3f}, "
          f"p = {wald_result['uncorrected'][1]:.3f} "
          f"(G = {wald_result['clusters']}, c = {wald_result['correction_factor']:.5f})")


if __name__ == "__main__":
    main()
