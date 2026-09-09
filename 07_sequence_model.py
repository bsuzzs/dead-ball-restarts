"""Step 7 (Appendix A) - post-corner sequence model benchmark.

A GRU over the events that follow a corner reaches a much higher AUC than a
tabular model using only the delivery endpoint and match covariates. That gap
is NOT evidence of incremental validity or of an "execution quality" signal:
the post-corner event sequence is downstream of whether the delivery reached
a dangerous area, so it is a mediator. Conditioning on a mediator inflates
apparent predictive power by construction.

Shot events are stripped from the sequence so the label cannot leak directly.
"""
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from common import read_restarts, write_json
from config import (LOOSE_WINDOW, PERIOD_OFFSET, SEED, SELF_SHOT_SUBEVENTS,
                    WYSCOUT_EVENTS)

MAX_STEPS = 12


def is_shot(event) -> bool:
    return event["eventName"] == "Shot" or (
        event["eventName"] == "Free Kick"
        and event.get("subEventName") in SELF_SHOT_SUBEVENTS)


def collect_sequences(corner_ids):
    sequences = {}
    for path in sorted(WYSCOUT_EVENTS.glob("events_*.json")):
        events = json.load(path.open())
        by_match = defaultdict(list)
        for event in events:
            if event["matchPeriod"] in PERIOD_OFFSET:
                by_match[event["matchId"]].append(event)
        for match_events in by_match.values():
            match_events.sort(key=lambda e: (PERIOD_OFFSET[e["matchPeriod"]], e["eventSec"]))
            for i, event in enumerate(match_events):
                if event["id"] not in corner_ids:
                    continue
                steps = []
                for later in match_events[i + 1:]:
                    if later["matchPeriod"] != event["matchPeriod"]:
                        break
                    if later["eventSec"] - event["eventSec"] > LOOSE_WINDOW:
                        break
                    if is_shot(later):
                        continue  # the label must not enter the sequence
                    position = later.get("positions", [{}])
                    steps.append((later["eventName"],
                                  int(later["teamId"] == event["teamId"]),
                                  position[0].get("x", 50) / 100,
                                  position[0].get("y", 50) / 100,
                                  min((later["eventSec"] - event["eventSec"]) / LOOSE_WINDOW, 1.0)))
                    if len(steps) >= MAX_STEPS:
                        break
                sequences[event["id"]] = steps
        print(f"  {path.stem}: {len(sequences):,} corner sequences")
        del events, by_match
    return sequences


def main() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch is not installed; skipping the Appendix A benchmark")
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    restarts = read_restarts()
    corners = restarts[restarts.restart == "Corner"].copy()
    sequences = collect_sequences(set(corners.eventId))

    vocabulary = sorted({step[0] for steps in sequences.values() for step in steps})
    index = {name: i + 1 for i, name in enumerate(vocabulary)}   # 0 is padding
    tokens = np.zeros((len(corners), MAX_STEPS), dtype=np.int64)
    numeric = np.zeros((len(corners), MAX_STEPS, 4), dtype=np.float32)
    for row, event_id in enumerate(corners.eventId):
        for step, values in enumerate(sequences.get(event_id, [])[:MAX_STEPS]):
            tokens[row, step] = index[values[0]]
            numeric[row, step] = values[1:]

    y = corners.shot.to_numpy().astype(np.float32)
    groups = corners.matchId.to_numpy()
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                         random_state=SEED).split(tokens, groups=groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("match leakage between train and test")

    class SequenceModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size + 1, 16, padding_idx=0)
            self.gru = nn.GRU(20, 48, batch_first=True)
            self.head = nn.Linear(48, 1)

        def forward(self, token_ids, features):
            hidden, _ = self.gru(torch.cat([self.embedding(token_ids), features], -1))
            return self.head(hidden[:, -1]).squeeze(-1)

    model = SequenceModel(len(vocabulary))
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    token_tensor = torch.tensor(tokens)
    numeric_tensor = torch.tensor(numeric)
    label_tensor = torch.tensor(y)

    for epoch in range(25):
        model.train()
        order = np.random.permutation(train)
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            optimiser.zero_grad()
            loss = loss_fn(model(token_tensor[batch], numeric_tensor[batch]),
                           label_tensor[batch])
            loss.backward()
            optimiser.step()
        if epoch % 8 == 7:
            model.eval()
            with torch.no_grad():
                auc = roc_auc_score(y[test], model(token_tensor[test],
                                                   numeric_tensor[test]).numpy())
            print(f"  epoch {epoch}: test AUC = {auc:.3f}")

    model.eval()
    with torch.no_grad():
        gru_auc = float(roc_auc_score(y[test], model(token_tensor[test],
                                                     numeric_tensor[test]).numpy()))

    # Comparator on the SAME corners and the SAME split: delivery endpoint and
    # match covariates only, i.e. information available before the sequence.
    tabular = pd.get_dummies(corners[["end_x", "end_y", "x", "y", "minute",
                                      "second_half", "home", "score_state"]],
                             columns=["score_state"], dtype=float)
    tabular = tabular.fillna(tabular.median(numeric_only=True))
    boosted = HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                             learning_rate=0.06).fit(tabular.iloc[train], y[train])
    tabular_auc = float(roc_auc_score(y[test], boosted.predict_proba(tabular.iloc[test])[:, 1]))

    write_json(dict(corner_n=len(corners), test_n=int(len(test)),
                    test_matches=int(len(set(groups[test]))),
                    gru_auc=gru_auc, tabular_auc=tabular_auc,
                    vocabulary=len(vocabulary), max_steps=MAX_STEPS,
                    mean_steps=float(np.mean([len(sequences.get(e, []))
                                              for e in corners.eventId])),
                    interpretation="Mediator conditioning; not incremental validity."),
               "sequence_model.json")
    print(f"GRU AUC = {gru_auc:.3f}  tabular AUC = {tabular_auc:.3f}")


if __name__ == "__main__":
    main()
