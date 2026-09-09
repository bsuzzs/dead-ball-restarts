"""Step 6 - supplementary predictive checks (diagnostic only).

These probe how much outcome-relevant information the tabular feature set
carries. They are not a predictive claim, and the SHAP ordering is not a
recruitment or training target: distance to goal dominates largely because
seven restart categories with very different pitch locations are pooled into
one model.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import read_restarts, write_csv, write_json
from config import PITCH_LENGTH_M, PITCH_WIDTH_M, SEED

NATIONAL_COMPETITIONS = {"European_Championship", "World_Cup"}
FEATURES = ["dist_goal", "lateral", "x", "y", "minute", "second_half", "home", "national"]


def build_features(restarts):
    frame = restarts.copy()
    frame["dist_goal"] = np.hypot((100 - frame.x) * PITCH_LENGTH_M / 100,
                                  (frame.y - 50) * PITCH_WIDTH_M / 100)
    frame["lateral"] = (frame.y - 50).abs()
    frame["national"] = frame.competition.isin(NATIONAL_COMPETITIONS).astype(int)
    x = pd.get_dummies(frame[FEATURES + ["restart", "score_state"]],
                       columns=["restart", "score_state"], dtype=float)
    return x.fillna(x.median(numeric_only=True)), frame


def main() -> None:
    restarts = read_restarts()
    x, frame = build_features(restarts)
    groups = frame.matchId.to_numpy()

    # Grouping by match keeps every restart from one match on one side of the
    # split; without it, within-match similarity leaks into the test score.
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                         random_state=SEED).split(x, groups=groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("match leakage between train and test")

    result = dict(train_n=int(len(train)), test_n=int(len(test)),
                  train_matches=int(len(set(groups[train]))),
                  test_matches=int(len(set(groups[test]))))
    models = {}
    for outcome in ["shot", "goal"]:
        y = frame[outcome].to_numpy()
        logistic = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=4000)).fit(x.iloc[train], y[train])
        boosted = HistGradientBoostingClassifier(
            random_state=SEED, max_iter=300, learning_rate=0.06,
            early_stopping=True, validation_fraction=0.15).fit(x.iloc[train], y[train])
        result[f"auc_logistic_{outcome}"] = float(
            roc_auc_score(y[test], logistic.predict_proba(x.iloc[test])[:, 1]))
        result[f"auc_boosted_{outcome}"] = float(
            roc_auc_score(y[test], boosted.predict_proba(x.iloc[test])[:, 1]))
        models[outcome] = boosted
        print(f"  {outcome}: logistic AUC = {result[f'auc_logistic_{outcome}']:.3f}, "
              f"boosted AUC = {result[f'auc_boosted_{outcome}']:.3f}")

    try:
        import shap
    except ImportError:
        print("  shap not installed; skipping the SHAP ordering")
        write_json(result, "predictive_checks.json")
        return

    rng = np.random.default_rng(SEED)
    sample = rng.choice(test, size=min(2500, len(test)), replace=False)
    background = shap.utils.sample(x.iloc[train], 100, random_state=0)
    for outcome, model in models.items():
        explainer = shap.PermutationExplainer(
            lambda z, model=model: model.predict_proba(z)[:, 1], background, seed=0)
        values = explainer(x.iloc[sample], max_evals=2 * x.shape[1] + 1,
                           silent=True)
        importance = (pd.Series(np.abs(values.values).mean(0), index=x.columns)
                      .sort_values(ascending=False))
        write_csv(importance.rename("mean_abs_shap").reset_index()
                  .rename(columns={"index": "feature"}), f"shap_importance_{outcome}.csv")
        result[f"shap_order_{outcome}"] = importance.index[:10].tolist()
    write_json(result, "predictive_checks.json")


if __name__ == "__main__":
    main()
