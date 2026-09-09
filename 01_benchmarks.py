"""Step 1 - seven-category benchmarks, window sensitivity and pitch geometry.

Produces the numbers behind Table 2, Figure 1, and the supplementary tables on
outcome windows, corner landing zones, direct free-kick distance bands and the
throw-in longitudinal gradient.
"""
import numpy as np
import pandas as pd

from common import MatchBootstrap, read_restarts, write_csv, write_json
from config import (BOX_X, BOX_Y, PITCH_LENGTH_M, PITCH_WIDTH_M,
                    RESTART_LABELS, RESTART_SUBEVENTS)

WINDOWS = (10, 15, 20)


def benchmarks(restarts, boot):
    rows = []
    for subtype in RESTART_SUBEVENTS:
        subset = restarts[restarts.restart == subtype]
        row = dict(restart=subtype, label=RESTART_LABELS[subtype], n=len(subset))
        for outcome in ["shot", "goal", "strict_shot", "strict_goal"]:
            stat = boot.rate(subset, outcome)
            row[f"{outcome}_events"] = stat["events"]
            row[f"{outcome}_pct"] = stat["rate"] * 100
            row[f"{outcome}_lo"] = stat["lo"] * 100
            row[f"{outcome}_hi"] = stat["hi"] * 100
        rows.append(row)
    return pd.DataFrame(rows)


def window_sensitivity(restarts):
    """10/15/20-s variants derived from the recorded first-shot time.

    Values are returned at full precision. Writing a rounded table and then
    formatting it again for display rounds twice, which can move a printed
    last digit (2.4945 -> 2.495 -> "2.50" instead of "2.49"). Round once, at
    the point of display.
    """
    rows = []
    for subtype in RESTART_SUBEVENTS:
        subset = restarts[restarts.restart == subtype]
        row = dict(restart=subtype, label=RESTART_LABELS[subtype], n=len(subset))
        for window in WINDOWS:
            hit = (subset.first_shot_sec <= window).fillna(False)
            row[f"shot{window}"] = hit.mean() * 100
            row[f"goal{window}"] = (hit & subset.first_shot_goal.astype(bool)).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def corner_zones(restarts, boot):
    """Landing-zone rates.

    Reported at zone level on purpose: individual hexagonal bins reach much
    higher values, but those peaks rest on very few corners and should not be
    read as a stable target area.

    The denominator here is EVERY corner with a recorded endpoint. The landing
    map in Figure 3 shows only endpoints with x >= 60, which is a smaller set.
    The two must not be quoted with the same n, so the totals are written out
    alongside the table.
    """
    corners = restarts[restarts.restart == "Corner"].dropna(subset=["end_x", "end_y"])
    inside_box = (corners.end_x >= BOX_X) & corners.end_y.between(*BOX_Y)
    central = corners.end_y.between(37, 63)
    zones = {
        "six_yard_box": (corners.end_x >= 94) & central,
        "six_to_penalty_spot": corners.end_x.between(89, 94, inclusive="left") & central,
        "penalty_spot_to_box_edge": corners.end_x.between(BOX_X, 89, inclusive="left") & central,
        "box_wide": inside_box & ~central,
        "outside_box": ~inside_box,
    }
    rows = []
    for name, mask in zones.items():
        subset = corners[mask]
        goal, shot = boot.rate(subset, "goal"), boot.rate(subset, "shot")
        rows.append(dict(zone=name, n=len(subset), goals=goal["events"],
                         goal_pct=goal["rate"] * 100, goal_lo=goal["lo"] * 100,
                         goal_hi=goal["hi"] * 100, shot_pct=shot["rate"] * 100))
    return pd.DataFrame(rows), len(corners)


def free_kick_distance(restarts, boot):
    """Two distance metrics, reported separately because they are not comparable.

    d_line is longitudinal distance to the goal line; d_centre is Euclidean
    distance to the centre of the goal. Quoting a band from one and comparing
    it with the other overstates or understates the gradient.
    """
    shots = restarts[restarts.restart == "Free kick shot"].dropna(subset=["x", "y"]).copy()
    shots["d_line"] = (100 - shots.x) * PITCH_LENGTH_M / 100
    shots["d_centre"] = np.hypot((100 - shots.x) * PITCH_LENGTH_M / 100,
                                 (shots.y - 50) * PITCH_WIDTH_M / 100)
    rows = []
    for metric in ["d_line", "d_centre"]:
        bands = pd.cut(shots[metric], [0, 22, 27, 32, np.inf], right=False)
        for band, subset in shots.groupby(bands, observed=True):
            stat = boot.rate(subset, "goal")
            rows.append(dict(metric=metric, band=str(band), n=len(subset),
                             goals=stat["events"], goal_pct=stat["rate"] * 100,
                             lo=stat["lo"] * 100, hi=stat["hi"] * 100))
    return pd.DataFrame(rows)


def throwin_gradient(restarts, boot):
    throws = restarts[restarts.restart == "Throw in"]
    coarse, fine = [], []
    for band, subset in throws.groupby(pd.cut(throws.x, np.arange(0, 101, 100 / 6),
                                              include_lowest=True), observed=True):
        shot, goal = boot.rate(subset, "shot"), boot.rate(subset, "goal")
        coarse.append(dict(band=str(band), lo_x=max(band.left, 0), hi_x=band.right,
                           n=len(subset), shot_pct=shot["rate"] * 100,
                           shot_lo=shot["lo"] * 100, shot_hi=shot["hi"] * 100,
                           goal_pct=goal["rate"] * 100, goal_lo=goal["lo"] * 100,
                           goal_hi=goal["hi"] * 100))
    for band, subset in throws.groupby(pd.cut(throws.x, np.arange(0, 101, 5),
                                              include_lowest=True), observed=True):
        if len(subset) >= 200:
            shot = boot.rate(subset, "shot")
            fine.append(dict(mid=(max(band.left, 0) + band.right) / 2, n=len(subset),
                             shot_pct=shot["rate"] * 100, lo=shot["lo"] * 100,
                             hi=shot["hi"] * 100))
    return pd.DataFrame(coarse), pd.DataFrame(fine)


def main() -> None:
    restarts = read_restarts()
    boot = MatchBootstrap(restarts.matchId)

    print("seven-category benchmarks ...")
    bench = benchmarks(restarts, boot)
    write_csv(bench, "benchmarks.csv")

    print("outcome-window sensitivity ...")
    windows = window_sensitivity(restarts)
    write_csv(windows, "window_sensitivity.csv")
    # The 15-s column of the window table and the headline benchmark table are
    # the same quantity computed two ways; if they ever diverge, one of the two
    # is wrong and any supplement built from both would disagree with itself.
    merged = bench.merge(windows, on="restart", suffixes=("_b", "_w"))
    for outcome in ("shot", "goal"):
        gap = (merged[f"{outcome}_pct"] - merged[f"{outcome}15"]).abs().max()
        if gap > 1e-9:
            raise AssertionError(f"15-s window disagrees with the benchmark "
                                 f"table for {outcome}: max gap {gap}")

    print("corner landing zones ...")
    zones, n_corners = corner_zones(restarts, boot)
    write_csv(zones, "corner_zones.csv")
    if int(zones.n.sum()) != n_corners:
        raise AssertionError("zone counts do not partition the corners")
    write_json(dict(corners_total=int((restarts.restart == "Corner").sum()),
                    corners_with_endpoint=int(n_corners),
                    corners_shown_in_map=int(((restarts.restart == "Corner")
                                              & (restarts.end_x >= 60)).sum()),
                    note="The zone table uses corners_with_endpoint; Figure 3 "
                         "shows corners_shown_in_map. Do not quote one for the other."),
              "corner_denominators.json")
    print(f"  corners with a recorded endpoint: {n_corners:,} "
          f"(landing map shows the x >= 60 subset only)")

    print("direct free-kick distance bands ...")
    write_csv(free_kick_distance(restarts, boot), "freekick_distance_bands.csv")

    print("throw-in longitudinal gradient ...")
    coarse, fine = throwin_gradient(restarts, boot)
    write_csv(coarse, "throwin_x_bands.csv")
    write_csv(fine, "throwin_x_fine.csv")

    # Per-competition corner rates, used for the external comparison table.
    corners = restarts[restarts.restart == "Corner"]
    rows = []
    for competition, subset in corners.groupby("competition"):
        shot, goal = boot.rate(subset, "shot"), boot.rate(subset, "goal")
        rows.append(dict(competition=competition, matches=subset.matchId.nunique(),
                         n=len(subset), shot_pct=shot["rate"] * 100,
                         shot_lo=shot["lo"] * 100, shot_hi=shot["hi"] * 100,
                         goal_pct=goal["rate"] * 100, goal_lo=goal["lo"] * 100,
                         goal_hi=goal["hi"] * 100))
    write_csv(pd.DataFrame(rows), "corner_by_competition.csv")


if __name__ == "__main__":
    main()
