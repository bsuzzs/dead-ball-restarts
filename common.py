"""Shared estimation machinery: match-cluster bootstrap, GEE, GLMM and matching.

All estimands here are observational associations. Rate differences and odds
ratios are on different scales and are not interchangeable.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from config import (CATEGORICAL_COVARIATES, N_BOOT, NUMERIC_COVARIATES,
                    OUTPUT_DIR, SEED)


# --------------------------------------------------------------------------
# Match-cluster bootstrap
# --------------------------------------------------------------------------
class MatchBootstrap:
    """Percentile bootstrap that resamples whole matches.

    One weight matrix is drawn once and reused for every quantity, so any two
    rates computed through this object are differenced *within* the same
    resample. That is what preserves the within-match correlation between
    groups, between outcome windows, and between own and opponent outcomes.
    """

    def __init__(self, match_ids, n_boot: int = N_BOOT, seed: int = SEED):
        self.match_ids = np.sort(np.unique(match_ids))
        rng = np.random.default_rng(seed)
        n = len(self.match_ids)
        self.weights = rng.multinomial(n, np.ones(n) / n, size=n_boot)

    def draws(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        """Bootstrap distribution of mean(column) over the given rows."""
        agg = (frame.groupby("matchId")[column].agg(["sum", "size"])
               .reindex(self.match_ids, fill_value=0))
        num = self.weights @ agg["sum"].to_numpy()
        den = self.weights @ agg["size"].to_numpy()
        return num / np.maximum(den, 1)

    def rate(self, frame: pd.DataFrame, column: str) -> dict:
        vals = self.draws(frame, column)
        lo, hi = np.quantile(vals, [0.025, 0.975])
        return dict(n=len(frame), events=int(frame[column].sum()),
                    rate=float(frame[column].mean()), lo=float(lo), hi=float(hi))

    def difference(self, treated: pd.DataFrame, control: pd.DataFrame,
                   column: str) -> dict:
        diff = self.draws(treated, column) - self.draws(control, column)
        lo, hi = np.quantile(diff, [0.025, 0.975])
        return dict(n1=len(treated), n0=len(control),
                    diff=float(treated[column].mean() - control[column].mean()),
                    lo=float(lo), hi=float(hi))


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}"


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------
def design_matrix(frame: pd.DataFrame, treatment: str | None = "tr",
                  drop: tuple = ()) -> pd.DataFrame:
    """Intercept, optional treatment, competition/score dummies, z-scored numerics.

    Constant columns are dropped rather than standardised, so subsets that
    contain a single competition still produce a full-rank design.

    `drop` removes numeric covariates. It exists for one specific situation:
    when the treatment is a deterministic function of a covariate, keeping that
    covariate in the model absorbs the very contrast being estimated. The
    throw-in zone indicator (x >= 67) is the case in point.
    """
    x = pd.get_dummies(frame[CATEGORICAL_COVARIATES], drop_first=True, dtype=float)
    for col in [c for c in NUMERIC_COVARIATES if c not in drop]:
        sd = frame[col].std()
        if sd > 1e-10:
            x[col] = (frame[col] - frame[col].mean()) / sd
    x.insert(0, "Intercept", 1.0)
    if treatment is not None:
        x.insert(1, "short", frame[treatment].astype(float).to_numpy())
    x = x.astype(float)
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError(f"rank-deficient design: {list(x.columns)}")
    return x


# --------------------------------------------------------------------------
# GEE: the primary adjusted model
# --------------------------------------------------------------------------
def fit_gee(frame: pd.DataFrame, outcome: str, label: str,
            treatment: str = "tr", drop: tuple = ()) -> dict:
    import statsmodels.api as sm

    x = design_matrix(frame, treatment, drop)
    model = sm.GEE(frame[outcome].to_numpy(float), x,
                   groups=frame.teamId.to_numpy(),
                   family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(maxiter=200, ctol=1e-8)
    if not result.converged:
        raise RuntimeError(f"GEE did not converge: {label}/{outcome}")
    ci = result.conf_int().loc["short"]
    return dict(analysis=label, model="GEE", outcome=outcome, n=len(frame),
                estimate=float(np.exp(result.params["short"])),
                lo=float(np.exp(ci.iloc[0])), hi=float(np.exp(ci.iloc[1])),
                p=float(result.pvalues["short"]),
                clusters=int(frame.teamId.nunique()),
                warnings="; ".join(str(w.message) for w in caught))


# --------------------------------------------------------------------------
# GLMM by maximum likelihood with Gauss-Hermite quadrature
# --------------------------------------------------------------------------
def fit_glmm(frame: pd.DataFrame, outcome: str, label: str,
             treatment: str = "tr", nodes=(60, 120)) -> dict:
    """Team random intercept, non-adaptive Gauss-Hermite quadrature.

    Variational approximations are avoided here: they understate posterior
    variance, and their output should not be reported as a confidence
    interval. Intervals below are Wald intervals from a numerical Hessian.
    """
    import statsmodels.api as sm
    from numpy.polynomial.hermite import hermgauss
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp

    frame = frame.sort_values("teamId", kind="stable")
    x = design_matrix(frame, treatment).to_numpy()
    y = frame[outcome].to_numpy(float)
    groups = frame.teamId.to_numpy()
    _, starts = np.unique(groups, return_index=True)
    codes = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(frame)]))

    start = np.r_[sm.GLM(y, x, family=sm.families.Binomial()).fit().params,
                  np.log(0.3)]
    runs, opt, objective = [], None, None
    for n_nodes in nodes:
        absc, weights = hermgauss(n_nodes)
        log_w = np.log(weights) - 0.5 * np.log(np.pi)

        def objective(par, absc=absc, log_w=log_w):
            random = np.sqrt(2) * np.exp(par[-1]) * absc
            eta = (x @ par[:-1])[:, None] + random[None, :]
            per_obs = y[:, None] * eta - np.logaddexp(0, eta)
            loglik = np.add.reduceat(per_obs, starts, axis=0) + log_w
            norm = logsumexp(loglik, axis=1)
            posterior = np.exp(loglik - norm[:, None])
            resid = (y[:, None] - expit(eta)) * posterior[codes]
            grad = np.r_[x.T @ resid.sum(axis=1), np.sum(resid * random)]
            return -norm.sum(), -grad

        opt = minimize(objective, start, jac=True, method="L-BFGS-B",
                       bounds=[(None, None)] * x.shape[1] + [(-7, 2)],
                       options={"maxiter": 350, "ftol": 1e-12, "gtol": 1e-6,
                                "maxls": 40})
        if not opt.success:
            raise RuntimeError(f"GLMM failed: {label}/{outcome}: {opt.message}")
        runs.append(dict(nodes=n_nodes, loglik=float(-opt.fun),
                         beta=float(opt.x[1]), sigma=float(np.exp(opt.x[-1]))))
        start = opt.x

    if abs(runs[0]["beta"] - runs[-1]["beta"]) >= 0.01:
        raise RuntimeError(f"quadrature unstable: {label}/{outcome}: {runs}")

    par, eps = opt.x, 1e-4
    hess = np.column_stack([
        (objective(par + np.eye(len(par))[j] * eps)[1]
         - objective(par - np.eye(len(par))[j] * eps)[1]) / (2 * eps)
        for j in range(len(par))])
    hess = (hess + hess.T) / 2
    if np.linalg.eigvalsh(hess).min() <= 0:
        raise RuntimeError(f"non-positive information: {label}/{outcome}")
    se = float(np.sqrt(np.linalg.inv(hess)[1, 1]))
    est = float(par[1])
    sigma = float(np.exp(par[-1]))
    return dict(analysis=label, model="GLMM_GH", outcome=outcome, n=len(frame),
                estimate=float(np.exp(est)),
                lo=float(np.exp(est - 1.96 * se)),
                hi=float(np.exp(est + 1.96 * se)),
                clusters=int(frame.teamId.nunique()), sigma=sigma,
                quadrature_beta_change=abs(runs[0]["beta"] - runs[-1]["beta"]),
                # A random-intercept SD at the optimiser's lower bound means the
                # model has collapsed to a pooled fit; agreement with GEE is then
                # not independent corroboration.
                sigma_at_bound=bool(sigma < np.exp(-6.5)))


# --------------------------------------------------------------------------
# Propensity-score matching
# --------------------------------------------------------------------------
def propensity_match(frame: pd.DataFrame, label: str, treatment: str = "tr"):
    """1:1 nearest neighbour without replacement, exact on competition.

    Both the caliper and the matching distance are on the logit scale. Mixing
    a logit-derived caliper with probability-scale distances silently changes
    which pairs are admissible, so the two must agree.

    Note the covariates are restart *starting* coordinates. Using the pass
    endpoint would make the endpoint-rule treatment a deterministic function
    of a covariate and violate positivity.
    """
    from sklearn.linear_model import LogisticRegression

    frame = frame.reset_index(drop=True).copy()
    x = design_matrix(frame, treatment=None).drop(columns="Intercept")
    prob = (LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
            .fit(x, frame[treatment]).predict_proba(x)[:, 1])
    clipped = np.clip(prob, 1e-8, 1 - 1e-8)
    logit = np.log(clipped / (1 - clipped))
    frame["ps"], frame["logit_ps"] = prob, logit

    pairs, support = [], []
    for competition, sub in frame.groupby("competition", sort=True):
        treated = sub[sub[treatment] == 1]
        control = sub[sub[treatment] == 0]
        if treated.empty or control.empty:
            continue
        lo = max(treated.logit_ps.min(), control.logit_ps.min())
        hi = min(treated.logit_ps.max(), control.logit_ps.max())
        caliper = 0.2 * sub.logit_ps.std(ddof=1)
        pool = list(control.index[control.logit_ps.between(lo, hi)])
        for i in treated.index[treated.logit_ps.between(lo, hi)]:
            if not pool:
                break
            distances = np.abs(logit[pool] - logit[i])
            k = int(np.argmin(distances))
            if distances[k] <= caliper:
                pairs.append((i, pool.pop(k)))
        support.append(dict(competition=competition, logit_overlap_lo=float(lo),
                            logit_overlap_hi=float(hi), caliper_logit=float(caliper),
                            treated=len(treated), control=len(control),
                            treated_in_support=int(treated.logit_ps.between(lo, hi).sum()),
                            control_in_support=int(control.logit_ps.between(lo, hi).sum())))

    treated_idx = [a for a, _ in pairs]
    control_idx = [b for _, b in pairs]
    if len(set(treated_idx)) != len(treated_idx) or len(set(control_idx)) != len(control_idx):
        raise RuntimeError("matching reused a unit; sampling was meant to be without replacement")

    # SMDs use the pre-matching pooled standard deviation as the denominator.
    denom = np.sqrt((x[frame[treatment] == 1].var(ddof=1)
                     + x[frame[treatment] == 0].var(ddof=1)) / 2)
    balance = pd.DataFrame({
        "variable": x.columns,
        "smd_before": ((x[frame[treatment] == 1].mean()
                        - x[frame[treatment] == 0].mean()) / denom).fillna(0).to_numpy(),
        "smd_after": ((x.loc[treated_idx].mean()
                       - x.loc[control_idx].mean()) / denom).fillna(0).to_numpy(),
    })
    frame["matched"] = False
    frame.loc[treated_idx + control_idx, "matched"] = True
    summary = dict(analysis=label, pairs=len(pairs),
                   max_abs_smd=float(balance.smd_after.abs().max()),
                   ps_min=float(prob.min()), ps_max=float(prob.max()))
    return frame.loc[treated_idx + control_idx], balance, pd.DataFrame(support), summary, frame


# --------------------------------------------------------------------------
# Small IO helpers
# --------------------------------------------------------------------------
def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT_DIR / name, index=False)
    print(f"  wrote {name}  ({len(frame)} rows)")


def write_json(obj, name: str) -> None:
    (OUTPUT_DIR / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                              default=str))
    print(f"  wrote {name}")


def read_restarts() -> pd.DataFrame:
    path = OUTPUT_DIR / "restarts.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(OUTPUT_DIR / "restarts.csv.gz")
