"""Withholding subtraces from a query answer (the partial-coverage ablation)."""
from expreuse.ablations import LEVELS, all_removals, nested_removals, removal_count, withhold
from expreuse.model import Experience, SubtraceHit


def _answer():
    hits = [[SubtraceHit("t", i, "alfworld", "p", "goal", 0.9, op, {}, [])] if op != "cool" else []
            for i, op in enumerate(("take", "cool", "put"))]
    return Experience("subtraces", subtraces=hits)


def test_levels_and_counts():
    assert LEVELS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert [removal_count(2, f) for f in LEVELS] == [0, 1, 1, 2, 2]
    assert [removal_count(4, f) for f in LEVELS] == [0, 1, 2, 3, 4]


def test_all_subsets_of_covered_positions():
    assert all_removals(_answer(), 0.5) == [(0,), (2,)]
    assert all_removals(_answer(), 1.0) == [(0, 2)]
    assert all_removals(_answer(), 0.0) == [()]


def test_nested_removals_are_nested_and_fixed_by_the_key():
    answer = _answer()
    low, high = nested_removals(answer, 0.5, "task 1"), nested_removals(answer, 1.0, "task 1")
    assert set(low) <= set(high) and len(high) == 2
    assert nested_removals(answer, 0.5, "task 1") == low


def test_withholding_leaves_positions_uncovered():
    out = withhold(_answer(), (0,))
    assert out.uncovered == [0, 1] and out.route == "subtraces"
    assert withhold(_answer(), (0, 2)).route == "none"
