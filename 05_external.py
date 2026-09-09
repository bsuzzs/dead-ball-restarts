"""Step 5 - external sources under a common rule where the source permits it.

Wyscout, StatsBomb and IMPECT are all reduced to the same event-level rule:
the first same-team shot within 15 seconds, in the same period. Understat is
shot-level with only a situation label, so no restart timeline can be
rebuilt from it and the 15-second rule does not apply.

Two provider traps are handled explicitly.

- IMPECT emits a separate GOAL event after a scoring SHOT. Counting both
  inflates the corner conversion rate (3.92% instead of 3.13%).
- Bootstrap intervals use a fresh generator per quantity, so a source's
  interval does not depend on the order in which quantities were computed.
"""
import json

import numpy as np
import pandas as pd

from common import write_csv
from config import (IMPECT_EVENTS, LOOSE_WINDOW, N_BOOT, SEED, STATSBOMB_DIR,
                    UNDERSTAT_DIR)

IMPECT_TYPES = {"CORNER": "corner", "THROW_IN": "throw_in",
                "GOAL_KICK": "goal_kick", "FREE_KICK": "free_kick"}
STATSBOMB_PASS = {"Corner": "corner", "Throw-in": "throw_in",
                  "Goal Kick": "goal_kick", "Free Kick": "free_kick_pass"}
STATSBOMB_SHOT = {"Free Kick": "free_kick_shot", "Penalty": "penalty"}
# The four tournaments that make up the 198-match xG subset.
STATSBOMB_TOURNAMENTS = ["Euro2020", "WorldCup2022", "Euro2024", "CopaAmerica2024"]


def cluster_ci(frame, key, column):
    """Match-cluster bootstrap with an independent generator per call."""
    rng = np.random.default_rng(SEED)
    agg = frame.groupby(key)[column].agg(["sum", "size"])
    n = len(agg)
    weights = rng.multinomial(n, np.ones(n) / n, size=N_BOOT)
    draws = (weights @ agg["sum"].to_numpy()) / np.maximum(weights @ agg["size"].to_numpy(), 1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return frame[column].mean() * 100, lo * 100, hi * 100


def impect_restarts():
    rows = []
    paths = sorted(IMPECT_EVENTS.glob("events_*.json"))
    for path in paths:
        events = sorted(json.load(path.open()), key=lambda e: e["index"])
        for i, event in enumerate(events):
            if event["actionType"] not in IMPECT_TYPES:
                continue
            start = event["gameTime"]["gameTimeInSec"]
            squad, period = event["squadId"], event["periodId"]
            first, later_goal = None, 0
            for other in events[i + 1:]:
                if other["periodId"] != period:
                    break
                if other["gameTime"]["gameTimeInSec"] - start > LOOSE_WINDOW:
                    break
                if other["squadId"] != squad:
                    continue
                if other["actionType"] in ("SHOT", "GOAL") and first is None:
                    first = other
                if other["actionType"] == "GOAL":
                    later_goal = 1
            shot = int(first is not None)
            goal = int(first is not None
                       and (first["actionType"] == "GOAL" or first.get("result") == "SUCCESS"))
            rows.append(dict(matchId=path.stem[len("events_"):],
                             restart=IMPECT_TYPES[event["actionType"]],
                             shot=shot, goal=goal,
                             goal_including_later=max(goal, later_goal)))
    return pd.DataFrame(rows), len(paths)


def statsbomb_restarts():
    def seconds(event):
        hours, minutes, secs = event["timestamp"].split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(secs)

    rows = []
    for tournament in STATSBOMB_TOURNAMENTS:
        folder = STATSBOMB_DIR / tournament / "events"
        for path in sorted(folder.glob("*.json")):
            events = sorted(json.load(path.open()), key=lambda e: e["index"])
            for i, event in enumerate(events):
                kind = None
                if event["type"]["name"] == "Pass":
                    kind = STATSBOMB_PASS.get(
                        event.get("pass", {}).get("type", {}).get("name"))
                elif event["type"]["name"] == "Shot":
                    kind = STATSBOMB_SHOT.get(
                        event.get("shot", {}).get("type", {}).get("name"))
                if kind is None or event["period"] > 4:
                    continue
                # Self-shot restarts use the restart shot only, including for xg_window.
                # This preserves the manuscript estimand; later rebound shots are excluded.
                if kind in ("free_kick_shot", "penalty"):
                    own_xg = float(event["shot"].get("statsbomb_xg", 0.0))
                    rows.append(dict(competition=tournament, matchId=path.stem,
                                     restart=kind, shot=1,
                                     goal=int(event["shot"]["outcome"]["name"] == "Goal"),
                                     xg_window=own_xg, xg_first_shot=own_xg))
                    continue
                start, squad = seconds(event), event["team"]["id"]
                shot = goal = 0
                # Two distinct quantities, kept separate on purpose:
                #   xg_window     - xG summed over EVERY same-team shot in the
                #                   15-s window (a restart that produces three
                #                   shots contributes all three);
                #   xg_first_shot - xG of the first such shot only, matching
                #                   the binary shot/goal outcomes.
                # The manuscript quotes the window sum averaged over restarts,
                # which is not an average xG per shot.
                xg_window = 0.0
                xg_first = 0.0
                for other in events[i + 1:]:
                    if other["period"] != event["period"]:
                        break
                    if seconds(other) - start > LOOSE_WINDOW:
                        break
                    if other["team"]["id"] == squad and other["type"]["name"] == "Shot":
                        value = float(other["shot"].get("statsbomb_xg", 0.0))
                        xg_window += value
                        if not shot:
                            shot = 1
                            goal = int(other["shot"]["outcome"]["name"] == "Goal")
                            xg_first = value
                rows.append(dict(competition=tournament, matchId=path.stem,
                                 restart=kind, shot=shot, goal=goal,
                                 xg_window=xg_window, xg_first_shot=xg_first))
    return pd.DataFrame(rows)


def understat_shots():
    rows = []
    for season in (2017, 2024):
        payload = json.load((UNDERSTAT_DIR / f"EPL_{season}_shots.json").open())
        if len(payload) != 380:
            raise AssertionError(f"expected 380 matches for {season}, got {len(payload)}")
        shots = [s for match in payload.values() for side in match.values() for s in side]
        if len({s["id"] for s in shots}) != len(shots):
            raise AssertionError("duplicate Understat shot ids")
        for situation in sorted({s["situation"] for s in shots}):
            subset = [s for s in shots if s["situation"] == situation]
            goals = sum(s["result"] == "Goal" for s in subset)
            expected = sum(float(s["xG"]) for s in subset)
            rows.append(dict(season=season, situation=situation, matches=len(payload),
                             shots=len(subset), goals=goals,
                             shots_per_match=len(subset) / len(payload),
                             goals_per_match=goals / len(payload),
                             conversion_pct=goals / len(subset) * 100,
                             xg=expected, xg_per_shot=expected / len(subset)))
    return pd.DataFrame(rows)


def main() -> None:
    print("IMPECT Bundesliga 2023/24 ...")
    impect, n_matches = impect_restarts()
    rows = []
    for restart, subset in impect.groupby("restart"):
        shot = cluster_ci(subset, "matchId", "shot")
        goal = cluster_ci(subset, "matchId", "goal")
        inflated = cluster_ci(subset, "matchId", "goal_including_later")
        rows.append(dict(source="IMPECT_Bundesliga_2324", restart=restart,
                         matches=n_matches, n=len(subset),
                         shot_pct=shot[0], shot_lo=shot[1], shot_hi=shot[2],
                         goal_pct=goal[0], goal_lo=goal[1], goal_hi=goal[2],
                         goal_pct_including_later=inflated[0]))
    write_csv(pd.DataFrame(rows), "impect_rates.csv")
    write_csv(impect, "impect_restarts.csv")

    print("StatsBomb tournaments ...")
    statsbomb = statsbomb_restarts()
    if statsbomb.matchId.nunique() != 198:
        raise AssertionError(f"expected 198 StatsBomb matches, "
                             f"got {statsbomb.matchId.nunique()}")
    corner_rows = []
    corners = statsbomb[statsbomb.restart == "corner"]
    for competition, subset in corners.groupby("competition"):
        shot = cluster_ci(subset, "matchId", "shot")
        goal = cluster_ci(subset, "matchId", "goal")
        corner_rows.append(dict(source=competition, matches=subset.matchId.nunique(),
                                n=len(subset), shot_pct=shot[0], shot_lo=shot[1],
                                shot_hi=shot[2], goal_pct=goal[0], goal_lo=goal[1],
                                goal_hi=goal[2]))
    write_csv(pd.DataFrame(corner_rows), "statsbomb_corner_rates.csv")

    xg_rows = []
    for restart, subset in statsbomb.groupby("restart"):
        goal = cluster_ci(subset, "matchId", "goal")
        with_shot = subset[subset.shot == 1]
        xg_rows.append(dict(
            restart=restart, n=len(subset),
            shot_pct=subset.shot.mean() * 100, goal_pct=goal[0],
            goal_lo=goal[1], goal_hi=goal[2],
            # Per restart, over all restarts of this type.
            # window fields: pass restarts sum 15-s shots; direct free kicks
            # and penalties include the restart shot only.
            mean_window_xg_per_restart=subset.xg_window.mean(),
            mean_first_shot_xg_per_restart=subset.xg_first_shot.mean(),
            # Conditional on the restart producing a shot at all.
            mean_window_xg_given_shot=with_shot.xg_window.mean() if len(with_shot) else float("nan"),
            mean_first_shot_xg_given_shot=with_shot.xg_first_shot.mean() if len(with_shot) else float("nan")))
    write_csv(pd.DataFrame(xg_rows).sort_values("mean_window_xg_per_restart",
                                                ascending=False),
              "statsbomb_xg_by_restart.csv")
    penalties = (statsbomb[statsbomb.restart == "penalty"]
                 .groupby("competition").goal.agg(["size", "mean"]).reset_index())
    penalties["conversion_pct"] = penalties["mean"] * 100
    write_csv(penalties, "statsbomb_penalties.csv")
    write_csv(statsbomb, "statsbomb_restarts.csv")

    print("Understat Premier League ...")
    write_csv(understat_shots(), "understat_shots.csv")


if __name__ == "__main__":
    main()
