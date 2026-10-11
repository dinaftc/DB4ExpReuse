"""The experiment grid: every condition has a distinct name, the command line builds the same conditions,
and the store files follow the naming the producers' collection writes."""
from expreuse import config
from expreuse.experiments import run
from expreuse.experiments.grid import EXPERIMENTS, Condition, cross_application, same_application


def test_conditions_are_distinct_and_counted():
    names = [c.name() for f in EXPERIMENTS.values() for c in f()]
    assert len(names) == len(set(names))
    counts = {name: sum(1 for _ in f()) for name, f in EXPERIMENTS.items()}
    assert counts == {'collect_stores': 6, 'expert_store': 1, 'same_application': 40, 'cross_agent': 18, 'cross_application': 22, 'theta': 20, 'depth': 22, 'coverage': 24, 'grounding': 15}


def test_names_and_stores():
    c = Condition("x", "alfworld", "dagger", "whole", source="gemma")
    assert c.name() == "alfworld_dagger_whole_from_gemma" and c.producer == "Gemma-26B"
    assert str(c.store_file("stores")) == "stores/alfworld_gemma.jsonl"
    c = next(c for c in cross_application() if c.cross_application and c.consumer == "dagger")
    assert c.name() == "alfworld_dagger_whole_from_scienceworld_gemma_cross_application"
    assert str(c.store_file("stores")) == "stores/scienceworld_gemma.jsonl"
    noplan = next(c for c in same_application() if c.hide_plan)
    assert noplan.name() == "alfworld_gemma_none_noplan" and noplan.ground
    collect = next(c for c in EXPERIMENTS["collect_stores"]())
    assert collect.name() == "alfworld_gemma_collect" and collect.hide_plan
    assert collect.task_list() == config.APPLICATIONS["alfworld"]["store_tasks"]
    expert = next(c for c in EXPERIMENTS["expert_store"]())
    assert expert.task_list() == "all" and expert.bindings == "none" and expert.producer == "expert"
    ungrounded = [c.name() for c in EXPERIMENTS["grounding"]() if c.hide_plan][0]
    assert ungrounded == "alfworld_gemma_whole_noplan_ungrounded"


def test_command_line_builds_the_same_condition():
    args = run.parse(["--application", "scienceworld", "--consumer", "imitation", "--mode", "subtraces",
                      "--withhold", "0.5", "--withhold-schedule", "nested", "--k", "2"])
    assert run.condition_from(run.parse(["--application", "alfworld", "--consumer", "expert", "--tasks", "all",
                                         "--bindings", "none", "--collect"])).name() == "alfworld_expert_collect"
    c = run.condition_from(args)
    assert c == Condition("run", "scienceworld", "imitation", "subtraces", withhold=0.5, schedule="nested", k=2)
    assert c.name() == "scienceworld_imitation_subtraces_k2_withhold0.5"
    assert run.settings_from(args).runs is None
