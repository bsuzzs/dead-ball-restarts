"""Step 0 - build the restart table from raw Wyscout event JSON.

Three things in here are easy to get wrong and are handled explicitly.

1. Period boundaries. Wyscout restarts the clock each period. Concatenating
   periods with a fixed offset interleaves first-half stoppage time with the
   start of the second half, which lets a 15-second outcome window straddle
   the interval. Events are therefore ordered by (period, within-period time)
   and no window may cross a period.

2. Goals. Tag 101 marks goals *and* goalkeeper save attempts, so a naive
   extraction credits saved shots to the attacking team. Goals are counted
   only on shot-like events; own goals (tag 102) are credited to the opponent.
   Every match is reconciled against the official score in the metadata.

3. Team strength. Points per game use only earlier matches in the same
   competition, so a team's final league position cannot leak backwards into
   its early-season restarts.

Outputs: restarts, team_matches, score_reconciliation, goal_attribution,
source_manifest, software_versions.
"""
from collections import Counter, defaultdict
import hashlib
import importlib.metadata as metadata
import json
import sys

import numpy as np
import pandas as pd

from common import write_csv, write_json
from config import (BOX_X, BOX_Y, EXPECTED_MATCHES, EXPECTED_RESTARTS,
                    EXPECTED_TEAMS, LOOSE_WINDOW, OPPONENT_GOAL_WINDOW,
                    OPPONENT_SHOT_WINDOW, OUTPUT_DIR, PERIOD_OFFSET,
                    RESTART_SUBEVENTS, SELF_SHOT_SUBEVENTS, SEQUENCE_WINDOW,
                    WYSCOUT_EVENTS, WYSCOUT_MATCHES)

SHORT_PASS_SUBTYPES = {"Simple pass", "Smart pass", ""}
# Widest outcome window kept, so 10/15/20-s variants need no rescan.
WIDEST_WINDOW = 20


def tags(event) -> set:
    return {t["id"] for t in event.get("tags", [])}


def is_shot(event) -> bool:
    return event["eventName"] == "Shot" or (
        event["eventName"] == "Free Kick"
        and event.get("subEventName") in SELF_SHOT_SUBEVENTS)


def is_goal(event) -> bool:
    """A goal is a shot-like event tagged 101 and not tagged 102 (own goal)."""
    return is_shot(event) and 101 in tags(event) and 102 not in tags(event)


def official_score(match, team_id) -> int:
    side = match["teamsData"][str(team_id)]
    return side["score"] if match["duration"] == "Regular" else side["scoreET"]


def load_matches():
    """Match metadata plus pre-match strength proxies, in chronological order."""
    matches, rows, strength = {}, [], {}
    for path in sorted(WYSCOUT_MATCHES.glob("matches_*.json")):
        competition = path.stem[len("matches_"):]
        points, played = defaultdict(float), defaultdict(int)
        for match in sorted(json.load(path.open()), key=lambda m: (m["dateutc"], m["wyId"])):
            match_id = match["wyId"]
            matches[match_id] = match
            teams = list(map(int, match["teamsData"]))
            if len(teams) != 2:
                raise ValueError(f"match {match_id} does not have two teams")
            for team in teams:
                opponent = next(t for t in teams if t != team)
                # Initial value 1.0 for a team's first appearance, with a zero
                # prior-match count so the model can down-weight it.
                strength[match_id, team] = dict(
                    team_ppg=points[team] / played[team] if played[team] else 1.0,
                    opp_ppg=points[opponent] / played[opponent] if played[opponent] else 1.0,
                    team_prior_games=played[team], opp_prior_games=played[opponent])
                goals, against = official_score(match, team), official_score(match, opponent)
                rows.append(dict(
                    matchId=match_id, teamId=team, oppId=opponent,
                    competition=competition, date=match["dateutc"],
                    goals=goals, opp_goals=against,
                    result="W" if goals > against else "L" if goals < against else "D",
                    home=int(match["teamsData"][str(team)]["side"] == "home"),
                    **strength[match_id, team]))
            for team in teams:
                opponent = next(t for t in teams if t != team)
                goals, against = official_score(match, team), official_score(match, opponent)
                points[team] += 3 if goals > against else 1 if goals == against else 0
                played[team] += 1
    return matches, pd.DataFrame(rows), strength


def scan_match(events, match, competition, strength):
    """Walk one match once, emitting restart records and goal bookkeeping."""
    match_id = match["wyId"]
    teams = list(map(int, match["teamsData"]))
    opponent_of = {t: next(o for o in teams if o != t) for t in teams}
    running, naive, corrected = Counter(), Counter(), Counter()
    restart_goal_ids, own_goal_ids = set(), set()
    records = []

    if len({e["id"] for e in events}) != len(events):
        raise ValueError(f"duplicate event ids in match {match_id}")

    for i, event in enumerate(events):
        team = event["teamId"]
        subtype = event.get("subEventName", "")
        period, second = event["matchPeriod"], event["eventSec"]

        if 101 in tags(event):
            naive[team] += 1
        if is_goal(event):
            corrected[team] += 1
        if 102 in tags(event):
            corrected[opponent_of[team]] += 1
            own_goal_ids.add(event["id"])

        if event["eventName"] == "Free Kick" and subtype in RESTART_SUBEVENTS:
            positions = event.get("positions", [])
            x0, y0 = (positions[0]["x"], positions[0]["y"]) if positions else (np.nan, np.nan)
            x1, y1 = (positions[1]["x"], positions[1]["y"]) if len(positions) > 1 else (np.nan, np.nan)

            shot = goal = strict_shot = strict_goal = 0
            opp_shot30 = opp_goal60 = 0
            sequence, first_own_shot, stopped = "unclassified", None, False
            seen_own_event, first_opponent_shot = False, None

            first_shot_elapsed = np.nan
            first_shot_goal = 0
            if subtype in SELF_SHOT_SUBEVENTS:
                # The restart is the shot.
                shot = strict_shot = 1
                goal = strict_goal = int(is_goal(event))
                first_shot_elapsed, first_shot_goal = 0.0, goal

            for later in events[i + 1:]:
                if later["matchPeriod"] != period:
                    break
                elapsed = later["eventSec"] - second
                if elapsed > OPPONENT_GOAL_WINDOW:
                    break
                same_team = later["teamId"] == team

                # Sequence rule: the first same-team event within 6 s.
                if same_team and not seen_own_event and elapsed <= SEQUENCE_WINDOW:
                    seen_own_event = True
                    sequence = ("short" if later["eventName"] == "Pass"
                                and later.get("subEventName", "") in SHORT_PASS_SUBTYPES
                                else "direct")
                if not same_team:
                    stopped = True

                # The first own shot is searched out to the widest window so
                # that 10/15/20-s variants are derivable without a second pass.
                if (subtype not in SELF_SHOT_SUBEVENTS and elapsed <= WIDEST_WINDOW
                        and same_team and is_shot(later) and first_own_shot is None):
                    first_own_shot = later
                    first_shot_elapsed = float(elapsed)
                    first_shot_goal = int(is_goal(later))
                    if elapsed <= LOOSE_WINDOW:
                        shot, goal = 1, first_shot_goal
                        if not stopped:
                            strict_shot, strict_goal = 1, goal

                # Counter-attack cost: the FIRST opponent shot only. Longer
                # windows are used here because a counter needs time to develop.
                if not same_team and is_shot(later) and first_opponent_shot is None:
                    first_opponent_shot = later
                    opp_shot30 = int(elapsed <= OPPONENT_SHOT_WINDOW)
                    opp_goal60 = int(is_goal(later))

            # Attribute the goal to this restart for the match-result analysis.
            if (subtype not in SELF_SHOT_SUBEVENTS and first_own_shot is not None
                    and first_shot_elapsed <= LOOSE_WINDOW and is_goal(first_own_shot)):
                restart_goal_ids.add(first_own_shot["id"])
            if subtype == "Free kick shot" and is_goal(event):
                restart_goal_ids.add(event["id"])

            records.append(dict(
                eventId=event["id"], matchId=match_id, teamId=team,
                oppId=opponent_of[team], playerId=event.get("playerId"),
                competition=competition, period=period, sec=second,
                restart=subtype, x=x0, y=y0, end_x=x1, end_y=y1,
                minute=(PERIOD_OFFSET[period] + second) / 60,
                second_half=int(period != "1H"),
                score_state=("leading" if running[team] > running[opponent_of[team]]
                             else "trailing" if running[team] < running[opponent_of[team]]
                             else "level"),
                home=int(match["teamsData"][str(team)]["side"] == "home"),
                coordinate_short=int(not (x1 >= BOX_X and BOX_Y[0] <= y1 <= BOX_Y[1])),
                sequence=sequence, shot=shot, goal=goal,
                strict_shot=strict_shot, strict_goal=strict_goal,
                opp_shot30=opp_shot30, opp_goal60=opp_goal60,
                first_shot_sec=first_shot_elapsed, first_shot_goal=first_shot_goal,
                **strength[match_id, team]))

        if is_goal(event):
            running[team] += 1
        if 102 in tags(event):
            running[opponent_of[team]] += 1

    quality = dict(matchId=match_id, competition=competition,
                   duration=match["duration"],
                   raw_tag101_match=all(naive[t] == official_score(match, t) for t in teams),
                   corrected_match=all(corrected[t] == official_score(match, t) for t in teams))
    for j, team in enumerate(teams):
        quality.update({f"team{j}": team, f"official{j}": official_score(match, team),
                        f"naive{j}": naive[team], f"corrected{j}": corrected[team]})

    attribution = []
    for team in teams:
        scored = [e for e in events if e["teamId"] == team and is_goal(e)]
        penalties = sum(e.get("subEventName") == "Penalty" for e in scored)
        own = sum(e["id"] in own_goal_ids and e["teamId"] == opponent_of[team] for e in events)
        restart = sum(e["id"] in restart_goal_ids and e.get("subEventName") != "Penalty"
                      for e in scored)
        open_play = len(scored) - penalties - restart
        if restart + open_play + penalties + own != corrected[team]:
            raise ValueError(f"goal attribution does not balance in match {match_id}")
        attribution.append(dict(matchId=match_id, teamId=team, setpiece=restart,
                                openplay=open_play, penalty=penalties, owngoal=own))
    return records, quality, attribution


def main() -> None:
    print("loading match metadata ...")
    matches, team_matches, strength = load_matches()

    records, quality, attribution, hashes, event_counts = [], [], [], {}, {}
    subtypes = Counter()
    for path in sorted(WYSCOUT_EVENTS.glob("events_*.json")):
        with path.open("rb") as handle:
            hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        competition = path.stem[len("events_"):]
        events = json.load(path.open())
        event_counts[competition] = len(events)

        by_match = defaultdict(list)
        for event in events:
            if event["eventName"] == "Free Kick":
                subtypes[event.get("subEventName", "")] += 1
            if event["matchPeriod"] in PERIOD_OFFSET:
                by_match[event["matchId"]].append(event)

        for match_id, match_events in by_match.items():
            match_events.sort(key=lambda e: (PERIOD_OFFSET[e["matchPeriod"]], e["eventSec"]))
            got = scan_match(match_events, matches[match_id], competition, strength)
            records.extend(got[0])
            quality.append(got[1])
            attribution.extend(got[2])
        print(f"  {competition}: {len(events):,} events -> {len(records):,} restarts so far")
        del events, by_match

    restarts = pd.DataFrame(records)

    # Regression checks against the published sample.
    if len(restarts) != EXPECTED_RESTARTS:
        raise AssertionError(f"expected {EXPECTED_RESTARTS:,} restarts, got {len(restarts):,}")
    if not restarts.eventId.is_unique:
        raise AssertionError("event ids are not unique")
    if restarts.matchId.nunique() != EXPECTED_MATCHES:
        raise AssertionError(f"expected {EXPECTED_MATCHES} matches")
    if restarts.teamId.nunique() != EXPECTED_TEAMS:
        raise AssertionError(f"expected {EXPECTED_TEAMS} teams")
    if not (restarts.strict_shot <= restarts.shot).all():
        raise AssertionError("strict shot outcome exceeds the permissive one")
    if not (restarts.strict_goal <= restarts.goal).all():
        raise AssertionError("strict goal outcome exceeds the permissive one")
    if not (restarts.goal <= restarts.shot).all():
        raise AssertionError("a goal was recorded without a shot")
    for col in ["x", "y", "end_x", "end_y"]:
        if not restarts[col].dropna().between(0, 100).all():
            raise AssertionError(f"{col} outside the 0-100 coordinate range")
    derived = (restarts.first_shot_sec.to_numpy() <= LOOSE_WINDOW).astype(int)
    if not np.array_equal(derived, restarts.shot.to_numpy().astype(int)):
        raise AssertionError("derived 15-s window disagrees with the recorded outcome")

    try:
        restarts.to_parquet(OUTPUT_DIR / "restarts.parquet", index=False)
    except Exception:  # pyarrow is optional
        restarts.to_csv(OUTPUT_DIR / "restarts.csv.gz", index=False)
    write_csv(team_matches, "team_matches.csv")
    write_csv(pd.DataFrame(quality), "score_reconciliation.csv")
    write_csv(pd.DataFrame(attribution), "goal_attribution.csv")
    write_json(dict(hashes=hashes, event_counts=event_counts,
                    free_kick_subtypes=dict(subtypes)), "source_manifest.json")

    versions = {}
    for package in ["pandas", "numpy", "scipy", "statsmodels", "scikit-learn",
                    "shap", "torch", "matplotlib", "pyarrow"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    versions["python"] = sys.version.split()[0]
    write_json(versions, "software_versions.json")

    counts = pd.DataFrame(quality)
    print(f"\nrestarts: {len(restarts):,}")
    print(restarts.restart.value_counts().to_string())
    print(f"\nscore reconciliation: naive {int(counts.raw_tag101_match.sum())}/{len(counts):,}"
          f"  corrected {int(counts.corrected_match.sum()):,}/{len(counts):,}")


if __name__ == "__main__":
    main()
