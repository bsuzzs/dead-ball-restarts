"""Step 3 - throw-in geography.

The attacking-third contrast is reported, but it largely restates location:
the model can adjust for score, time and venue, yet not for distance to goal,
because distance is what defines the zone. The comparisons that carry tactical
information are the ones made *within* the attacking third.
"""
import numpy as np
import pandas as pd

from common import MatchBootstrap, fit_gee, read_restarts, write_csv
from config import (ATTACKING_THIRD_X, LONG_THROW_M, PITCH_LENGTH_M,
                    PITCH_WIDTH_M)

OUTCOMES = ["shot", "goal", "strict_shot", "strict_goal"]


def contrast_rows(frame, boot, label):
    rows = []
    treated, control = frame[frame.tr == 1], frame[frame.tr == 0]
    for outcome in OUTCOMES:
        stat = boot.difference(treated, control, outcome)
        rate_t, rate_c = boot.rate(treated, outcome), boot.rate(control, outcome)
        rows.append(dict(analysis=label, outcome=outcome,
                         pct_1=rate_t["rate"] * 100, pct_0=rate_c["rate"] * 100,
                         diff_pp=stat["diff"] * 100, lo=stat["lo"] * 100,
                         hi=stat["hi"] * 100, n1=stat["n1"], n0=stat["n0"]))
    return rows


def main() -> None:
    restarts = read_restarts()
    boot = MatchBootstrap(restarts.matchId)
    throws = restarts[restarts.restart == "Throw in"].copy()

    # Zone contrast, reported with the caveat above.
    throws["tr"] = (throws.x >= ATTACKING_THIRD_X).astype(int)
    rows = contrast_rows(throws, boot, "zone")
    # x and y are dropped here because the zone indicator IS x >= 67. Leaving
    # them in lets the model absorb the contrast it is meant to estimate, which
    # is also why this odds ratio should be read as a restatement of location
    # rather than as an independent tactical effect.
    models = [fit_gee(throws, outcome, "zone", drop=("x", "y"))
              for outcome in OUTCOMES]
    print("zone contrast (restates location):")
    for m in models:
        print(f"  GEE {m['outcome']}: OR = {m['estimate']:.2f} "
              f"[{m['lo']:.2f}, {m['hi']:.2f}]")

    # Within the attacking third: direction and recorded displacement.
    # These describe where the ball was recorded as going, not the throw's
    # trajectory and not the thrower's intention.
    inner = throws[throws.tr == 1].dropna(subset=["end_x", "end_y"]).copy()
    inner["dx"] = inner.end_x - inner.x
    inner["length_m"] = np.hypot(inner.dx * PITCH_LENGTH_M / 100,
                                 (inner.end_y - inner.y) * PITCH_WIDTH_M / 100)
    inner["forward"] = (inner.dx > 0).astype(int)
    inner["long"] = (inner.length_m >= LONG_THROW_M).astype(int)

    for variable in ["forward", "long"]:
        frame = inner.copy()
        frame["tr"] = frame[variable]
        label = f"attacking_third_{variable}"
        rows += contrast_rows(frame, boot, label)
        # Direction and length are fitted separately; their joint or independent
        # contributions are not identified here.
        models += [fit_gee(frame, outcome, label) for outcome in ["shot", "goal"]]
        print(f"{label}: n1 = {int(frame.tr.sum()):,}")

    # Descriptive length bands.
    band_rows = []
    for band, subset in inner.groupby(pd.cut(inner.length_m, [0, 10, 20, 30, np.inf],
                                             right=False), observed=True):
        if len(subset):
            shot, goal = boot.rate(subset, "shot"), boot.rate(subset, "goal")
            band_rows.append(dict(band=str(band), n=len(subset),
                                  shot_pct=shot["rate"] * 100,
                                  shot_lo=shot["lo"] * 100, shot_hi=shot["hi"] * 100,
                                  goal_pct=goal["rate"] * 100))

    write_csv(pd.DataFrame(rows), "throwin_contrasts.csv")
    write_csv(pd.DataFrame(models), "throwin_models.csv")
    write_csv(pd.DataFrame(band_rows), "throwin_length_bands.csv")
    write_csv(inner[["eventId", "matchId", "x", "y", "end_x", "end_y", "length_m",
                     "forward", "long", "shot", "goal", "strict_shot", "strict_goal"]],
              "attacking_third_throwins.csv")


if __name__ == "__main__":
    main()
