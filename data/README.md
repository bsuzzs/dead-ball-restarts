# Raw data

None of the four sources is redistributed here. Obtain each one from its
publisher, then point `RESTART_DATA` at a directory with the layout below.

```bash
export RESTART_DATA=/path/to/raw
```

```
$RESTART_DATA/
├── events/                        # Wyscout, one file per competition
│   ├── events_England.json
│   ├── events_European_Championship.json
│   ├── events_France.json
│   ├── events_Germany.json
│   ├── events_Italy.json
│   ├── events_Spain.json
│   └── events_World_Cup.json
├── matches/                       # Wyscout match metadata, same competitions
│   ├── matches_England.json
│   └── ...
├── statsbomb_recent/
│   ├── Euro2020/events/*.json     # 51 matches
│   ├── Euro2024/events/*.json     # 51 matches
│   ├── WorldCup2022/events/*.json # 64 matches
│   └── CopaAmerica2024/events/*.json  # 32 matches
├── impect/open-data-main/data/events/
│   └── events_<matchId>.json      # Bundesliga 2023/24, 306 matches
└── understat/
    ├── EPL_2017_shots.json
    └── EPL_2024_shots.json
```

## Sources and terms

| Source | Coverage | Where | Terms |
| --- | --- | --- | --- |
| **Wyscout** (primary) | 1,941 matches: five major European leagues 2017/18, UEFA Euro 2016, 2018 FIFA World Cup | Pappalardo et al. (2019), [figshare collection 4415000](https://figshare.com/collections/Soccer_match_event_dataset/4415000) | CC BY 4.0 |
| **StatsBomb** | Euro 2020, World Cup 2022, Euro 2024, Copa América 2024 (198 matches, with xG) | StatsBomb Open Data | StatsBomb open-data licence; attribution required |
| **IMPECT** | Bundesliga 2023/24, 306 matches | IMPECT open-data repository | Commercial provider; redistribution restricted by their licence |
| **Understat** | Premier League 2017/18 and 2024/25 shot records | Public match pages | Public accessibility is not a redistribution grant; check the site's terms before reusing |

Wyscout is the only source used for the primary analysis. The other three
appear in the external-comparison section only.

## Notes that affect the numbers

- **Wyscout coordinates are normalised 0–100**, not metric. Conversions to
  metres in this code use a nominal 105 × 68 m pitch and are approximations.
- **The provider's `Free Kick` sub-event is a residual class** of free-kick
  passes. It is not equivalent to an indirect free kick as defined by the laws
  of the game, and is labelled "other free-kick pass" throughout.
- **IMPECT emits a separate `GOAL` event after a scoring `SHOT`.** Counting
  both inflates corner conversion (3.92% rather than 3.13% on the first-shot
  rule used here).
- **Understat is shot-level** with a `situation` label and no restart
  timeline, so the 15-second rule cannot be applied to it.
- **Penalty-shootout kicks (76) are excluded**; regulation and extra time are
  both retained.

## Expected sample after extraction

`00_extract.py` asserts these and fails loudly otherwise:

| Quantity | Value |
| --- | --- |
| Restarts | 193,197 |
| Matches | 1,941 |
| Teams | 142 |
| Corners | 19,316 |
| Other free-kick passes | 45,566 |

If your counts differ, the source files are a different vintage of the
dataset; do not silently relax the assertions.
