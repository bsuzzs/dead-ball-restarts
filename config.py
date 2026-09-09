"""Paths, constants and shared settings.

Point RESTART_DATA at the directory that holds the four raw sources. The
expected layout is documented in data/README.md. Nothing here writes to the
raw data; every script writes only under OUTPUT_DIR.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("RESTART_DATA", ROOT / "data" / "raw"))
OUTPUT_DIR = Path(os.environ.get("RESTART_OUT", ROOT / "outputs"))
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Raw source sub-paths.
WYSCOUT_EVENTS = DATA_DIR / "events"
WYSCOUT_MATCHES = DATA_DIR / "matches"
STATSBOMB_DIR = DATA_DIR / "statsbomb_recent"
IMPECT_EVENTS = DATA_DIR / "impect" / "open-data-main" / "data" / "events"
UNDERSTAT_DIR = DATA_DIR / "understat"

SEED = 20260908
N_BOOT = 2000

# Wyscout encodes each period with its own clock; these offsets order periods
# without ever letting an outcome window cross a period boundary.
PERIOD_OFFSET = {"1H": 0, "2H": 2700, "E1": 5400, "E2": 6300}

# The seven restart categories, as Wyscout Free Kick sub-events. "Free Kick"
# is the provider's residual free-kick pass class; it is NOT equivalent to an
# indirect free kick as defined by the laws of the game.
RESTART_SUBEVENTS = [
    "Corner", "Free kick cross", "Free kick shot",
    "Free Kick", "Throw in", "Goal kick", "Penalty",
]
# Restarts that are themselves a shot: shot rate is 100% by definition.
SELF_SHOT_SUBEVENTS = {"Free kick shot", "Penalty"}

RESTART_LABELS = {
    "Corner": "Corner",
    "Free kick cross": "Free-kick cross",
    "Free kick shot": "Direct free-kick shot",
    "Free Kick": "Other free-kick pass",
    "Throw in": "Throw-in",
    "Goal kick": "Goal kick",
    "Penalty": "Penalty",
}

# Outcome windows (seconds).
LOOSE_WINDOW = 15
OPPONENT_SHOT_WINDOW = 30
OPPONENT_GOAL_WINDOW = 60
SEQUENCE_WINDOW = 6

# Corner endpoint rule (sensitivity analysis). Wyscout uses 0-100 coordinates.
BOX_X = 83
BOX_Y = (21, 79)
BOX_X_THRESHOLDS = (80, 83, 86)

# Throw-in geography.
ATTACKING_THIRD_X = 67
LONG_THROW_M = 20

# Nominal pitch used to convert normalised coordinates to metres. This is an
# approximation: Wyscout coordinates are normalised, not metric.
PITCH_LENGTH_M = 105
PITCH_WIDTH_M = 68

# Model covariates shared by GEE, GLMM and the propensity model.
CATEGORICAL_COVARIATES = ["competition", "score_state"]
NUMERIC_COVARIATES = [
    "minute", "second_half", "home", "x", "y",
    "team_ppg", "opp_ppg", "team_prior_games", "opp_prior_games",
]

# Expected sample size, asserted after extraction as a regression check.
EXPECTED_RESTARTS = 193_197
EXPECTED_MATCHES = 1_941
EXPECTED_TEAMS = 142
