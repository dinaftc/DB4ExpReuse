"""The policy consumers' pure parts: gold-step matching and state text for the imitation policy, its scoring
head, and the DAgger policy's score normalisation."""
import random

import numpy as np
import pytest

from expreuse.policies.scienceworld_imitation import canonical, history_text, negatives


def test_gold_step_matches_the_valid_list():
    valid = {"pick up metal pot containing water", "focus on apple tree", "look at thermometer", "pick up thermometer"}
    assert canonical("pick up metal pot", valid) == "pick up metal pot containing water"
    assert canonical("focus on apple", valid) == "focus on apple tree"
    assert canonical("examine thermometer", valid) == "look at thermometer"
    assert canonical("pick up thermometer in inventory", valid) == "pick up thermometer"
    assert canonical("pick up", {"pick up a", "pick up b"}) is None


def test_negatives_prefer_confusable_actions():
    valid = ["pick up cup", "pick up jug", "move cup to table", "look around", "wait", "focus on cup"] + \
            [f"go to room {i}" for i in range(40)]
    chosen = negatives("pick up cup", valid, random.Random(0))
    assert "pick up cup" not in chosen
    assert {"pick up jug", "move cup to table", "focus on cup"} <= set(chosen)
    assert history_text(["a", "b", "c", "d"], "obs") == "b | c | d || obs"


def test_scoring_head_scores_every_candidate():
    torch = pytest.importorskip("torch")
    from expreuse.policies.scienceworld_imitation import make_head

    head = make_head(dim=8, hidden=16)
    state = head.states(torch.zeros(1, 32))[0]
    scores = head.score(state, head.actions(torch.randn(5, 8)))
    assert scores.shape == (5,)


def test_dagger_log_probs_normalise(monkeypatch):
    from expreuse.policies import alfworld_dagger

    class Fake(alfworld_dagger.DaggerPolicy):
        def __init__(self):
            self.temperature = 2.0

        def scores(self, history, admissible):
            return np.array([1.0, 3.0, 2.0])

    logp = np.asarray(Fake().log_probs(None, [("", "obs")], ["a", "b", "c"]))
    assert np.isclose(np.exp(logp).sum(), 1.0) and logp.argmax() == 1
