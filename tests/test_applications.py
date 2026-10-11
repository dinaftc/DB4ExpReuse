"""The two applications without their simulators: plans (from a game's PDDL, from a task description), step
labelling, segmentation of the sample traces, and the reuse queries over them."""
import csv
import json
from pathlib import Path

import pytest

from expreuse import ReuseRequest
from expreuse.embedding import HashEmbedder
from expreuse.envs.alfworld import (AlfworldLabeler, confirm_plans, entity_names, goal_entities, goal_operators,
                                    pddl_plan, task_parameters)
from expreuse.envs import alfworld as alfworld_module
from expreuse.envs.scienceworld import ScienceWorldLabeler, _room_contents, bind_plan, plan_operators, task_type
from expreuse.io import read_jsonl
from expreuse.model import PlanStep
from expreuse.operators import ALFWORLD, SCIENCEWORLD
from expreuse.segmentation import segment
from expreuse.stores import MemoryStore

DATA = Path(__file__).resolve().parents[1] / "data"
LIBRARIES = {"alfworld": ALFWORLD, "scienceworld": SCIENCEWORLD}

PROBLEM = """(define (problem p) (:domain alfred)
(:objects
agent1 - agent
Mug_bar__plus_00_dot_10_bar__plus_00_dot_90_bar__minus_00_dot_50 - object
Mug_bar__plus_00_dot_20_bar__plus_00_dot_90_bar__minus_00_dot_50 - object
CounterTop_bar__plus_00_dot_00_bar__plus_00_dot_95_bar__minus_00_dot_50 - receptacle
Microwave_bar__plus_01_dot_00_bar__plus_01_dot_50_bar__minus_00_dot_50 - receptacle
Cabinet_bar__plus_02_dot_00_bar__plus_00_dot_50_bar__minus_00_dot_50 - receptacle
Sink_bar__plus_03_dot_00_bar__plus_00_dot_90_bar__minus_00_dot_50_bar_SinkBasin - receptacle
)
(:init
(inReceptacle Mug_bar__plus_00_dot_20_bar__plus_00_dot_90_bar__minus_00_dot_50 CounterTop_bar__plus_00_dot_00_bar__plus_00_dot_95_bar__minus_00_dot_50)
)
(:goal (and)))"""
EXPERT = {
    "pddl_params": {"object_target": "Mug", "parent_target": "Cabinet", "toggle_target": "", "mrecep_target": ""},
    "plan": {"high_pddl": [
        {"planner_action": {"action": "GotoLocation"}},
        {"planner_action": {"action": "PickupObject", "objectId": "Mug|+00.20|+00.90|-00.50",
                            "coordinateReceptacleObjectId": ["CounterTop", []]}},
        {"planner_action": {"action": "HeatObject", "objectId": "Microwave|+01.00|+01.50|-00.50"}},
        {"planner_action": {"action": "PutObject", "objectId": "Mug|+00.20|+00.90|-00.50",
                            "receptacleObjectId": "Cabinet|+02.00|+00.50|-00.50"}},
        {"planner_action": {"action": "End"}}]}}


def test_alfworld_entity_names_follow_the_game():
    names = entity_names(PROBLEM)
    assert names["mug_bar__plus_00_dot_10_bar__plus_00_dot_90_bar__minus_00_dot_50"] == "mug 2"
    assert names["mug_bar__plus_00_dot_20_bar__plus_00_dot_90_bar__minus_00_dot_50"] == "mug 1"
    assert names["sink_bar__plus_03_dot_00_bar__plus_00_dot_90_bar__minus_00_dot_50_bar_sinkbasin"] == "sinkbasin 1"


def test_alfworld_plan_from_pddl(tmp_path):
    game = tmp_path / "game.tw-pddl"
    game.write_text(json.dumps({"pddl_problem": PROBLEM}))
    (tmp_path / "traj_data.json").write_text(json.dumps(EXPERT))
    assert pddl_plan(game) == (PlanStep("take", {"object": "mug 1", "source": "countertop 1"}),
                               PlanStep("heat", {"object": "mug 1", "appliance": "microwave 1"}),
                               PlanStep("put", {"object": "mug 1", "destination": "cabinet 1"}))
    assert task_parameters(game) == {"object": "mug", "destination": "cabinet", "appliance": ""}
    assert goal_entities("put a hot cup in cabinet") == ("cup", "cabinet")


def test_alfworld_plan_from_the_goal():
    goals = {"put some creditcard on desk.": ("take", "put"),
             "heat some mug and put it in coffeemachine.": ("take", "heat", "put"),
             "put a hot cup in cabinet": ("take", "heat", "put"),
             "cool some bread and put it in countertop.": ("take", "cool", "put"),
             "clean some soapbar and put it in garbagecan.": ("take", "clean", "put"),
             "put two book in drawer.": ("take", "put", "take", "put"),
             "find two cup and put them in microwave.": ("take", "put", "take", "put"),
             "look at bowl under the desklamp.": ("take", "use"),
             "examine the keychain with the desklamp.": ("take", "use")}
    for goal, operators in goals.items():
        assert goal_operators(goal) == operators, goal


def test_alfworld_bindings_from_the_scan():
    room = {"countertop 1": ["bread 1", "mug 2"], "microwave 1": [], "cabinet 1": [], "cabinet 2": ["mug 1"]}
    plan = alfworld_module.bind_plan(("take", "heat", "put"), "put a hot mug in cabinet", room)
    assert plan == (PlanStep("take", {"object": "mug 2", "source": "countertop 1"}),
                    PlanStep("heat", {"object": "mug 2", "appliance": "microwave 1"}),
                    PlanStep("put", {"object": "mug 2", "destination": "cabinet 1"}))
    unseen = alfworld_module.bind_plan(("take", "put"), "put a statue in coffeetable", {"shelf 1": []})
    assert unseen == (PlanStep("take"), PlanStep("put"))
    lamp = alfworld_module.bind_plan(("take", "use"), "look at bowl under the desklamp",
                                     {"desk 1": ["desklamp 1", "bowl 2"], "shelf 1": []})
    assert lamp == (PlanStep("take", {"object": "bowl 2", "source": "desk 1"}), PlanStep("use", {"appliance": "desklamp 1"}))


def test_alfworld_pddl_confirms_the_goal_plan(tmp_path, monkeypatch):
    (tmp_path / "game.tw-pddl").write_text(json.dumps({"pddl_problem": PROBLEM}))
    (tmp_path / "traj_data.json").write_text(json.dumps(EXPERT))
    monkeypatch.setattr(alfworld_module, "DATA", tmp_path)
    assert confirm_plans(["game.tw-pddl"], {"game.tw-pddl": "put a hot mug in cabinet"}) == []
    assert confirm_plans(["game.tw-pddl"], {"game.tw-pddl": "put a mug in cabinet"}) == [
        ("game.tw-pddl", ("take", "put"), ("take", "heat", "put"))]


def test_alfworld_labeler():
    labeler = AlfworldLabeler("put a hot cup in cabinet")
    actions = ["go to countertop 1", "take cup 1 from countertop 1", "put cup 1 in/on microwave 1",
               "heat cup 1 with microwave 1", "take cup 1 from microwave 1", "put cup 1 in/on cabinet 3"]
    assert [labeler.operator_of(a) for a in actions] == [None, "take", None, "heat", None, "put"]
    assert labeler.completes("heat", {}, "heat cup 1 with microwave 1", "You heat the cup 1 using the microwave 1.")
    assert not labeler.completes("heat", {}, "heat cup 1 with microwave 1", "Nothing happens.")
    assert labeler.binding("take", [("go to countertop 1", "ok"), ("take cup 1 from countertop 1", "ok")]) == \
        {"object": "cup 1", "source": "countertop 1"}
    assert AlfworldLabeler(destination="fridge").operator_of("put cup 1 in/on fridge 1") == "put"


def test_scienceworld_plan_from_description():
    paint = "Your task is to use chemistry to create violet paint. When you are done, focus on the violet paint."
    assert plan_operators(paint) == ("pour", "mix", "focus on")
    animal = "Your task is to find a(n) animal. First, focus on the thing. Then, move it to the orange box in the kitchen."
    assert task_type(animal) == "find-animal" and plan_operators(animal) == ("focus on", "take", "put")
    assert plan_operators("", "melt")[-1] == "measure"
    assert "heat" in plan_operators("", "boil") and "cool" in plan_operators("", "freeze")


def test_scienceworld_labeler():
    labeler = ScienceWorldLabeler()
    actions = ["teleport to kitchen", "pick up thermometer", "move lead to blast furnace", "activate blast furnace",
               "use thermometer on lead", "pour yellow paint into cup", "mix cup", "focus on violet paint",
               "move ice to freezer", "activate sink", "look around"]
    assert [labeler.operator_of(a) for a in actions] == \
        [None, "take", "put", "heat", "measure", "pour", "mix", "focus on", "cool", "activate", None]
    assert labeler.completes("take", {}, "pick up thermometer", "You move the thermometer to the inventory.")
    assert not labeler.completes("take", {}, "pick up thermometer", "No known action matches that input.")
    assert labeler.binding("take", [("teleport to kitchen", "ok"), ("pick up thermometer", "ok")]) == \
        {"object": "thermometer", "source": "kitchen"}
    assert labeler.binding("heat", [("move lead to blast furnace", "ok"), ("activate blast furnace", "ok")]) == \
        {"object": "lead", "appliance": "blast furnace"}
    assert labeler.binding("cool", [("move ice to freezer", "ok")]) == {"object": "ice", "appliance": "freezer"}


def test_scienceworld_bindings_from_the_scan():
    look = ("This room is called the kitchen. In it, you see: \n\tthe agent\n\ta substance called air\n"
            "\ta counter. On the counter is: a bowl (containing a red apple, a banana), a drawer.\n"
            "\ta freezer. The freezer door is closed. \n\ta glass jar (containing a substance called sodium chloride)\n"
            "\ta stove, which is turned off. On the stove is: nothing.\n"
            "\ta table. On the table is: a glass cup (containing a substance called water).\n"
            "\ta thermometer, currently reading a temperature of 10 degrees celsius\n"
            "You also see:\n\tA door to the hallway (that is open)\n")
    rooms = {"kitchen": _room_contents(look)}
    assert rooms["kitchen"] == ["counter", "red apple", "banana", "bowl", "drawer", "freezer", "glass jar",
                                "sodium chloride", "stove", "table", "water", "glass cup", "thermometer"]
    plan = bind_plan(("focus on", "cool", "heat", "measure"),
                     "Your task is to boil water. First, focus on the substance. Then, take actions that will "
                     "cause it to change its state of matter.", rooms)
    assert plan[0].binding == {"object": "water"}
    assert plan[1].binding == {"object": "water", "appliance": "freezer"}
    assert plan[2].binding == {"object": "water", "appliance": "stove"}
    assert plan[3].binding == {"object": "water", "instrument": "thermometer"}


@pytest.mark.parametrize("application", ["alfworld", "scienceworld"])
def test_sample_operators_have_signatures(application):
    operators = json.loads((DATA / application / "operators.json").read_text())
    for row in operators:
        assert row["operator"] in LIBRARIES[application]
    stored = {s["operator"] for s in csv.DictReader(open(DATA / application / "subtraces.csv"))}
    assert stored <= {row["operator"] for row in operators}


def _labeler(application, item):
    if application == "alfworld":
        return AlfworldLabeler(item.task.goal)
    return ScienceWorldLabeler()


@pytest.mark.parametrize("application", ["alfworld", "scienceworld"])
def test_sample_traces_segment_with_the_labeler(application):
    """Re-segmenting a sample trace with the application's labeler and its stored plan reproduces its stored
    steps and subtraces."""
    plans = {}
    for row in csv.DictReader(open(DATA / application / "plans.csv")):
        plans.setdefault(row["task_id"], []).append(PlanStep(row["operator"], json.loads(row["binding"])))
    for item in read_jsonl(DATA / application / "traces.jsonl"):
        raw = [(s.action, s.observation) for s in item.steps]
        steps, subtraces = segment(item.trace.trace_id, tuple(plans[item.task.task_id]), raw, _labeler(application, item))
        assert len(steps) == len(item.steps)
        assert subtraces == item.subtraces
        assert [s.subgoal_id for s in steps] == [s.subgoal_id for s in item.steps]


@pytest.mark.parametrize("application", ["alfworld", "scienceworld"])
def test_queries_over_the_sample(application):
    embedder = HashEmbedder(64)
    store = MemoryStore()
    items = list(read_jsonl(DATA / application / "traces.jsonl", embedder))
    for item in items:
        store.insert(item)
    producers = {i.trace.producer for i in items}
    assert len(producers) == 2 and store.counts()["traces"] == len(items)
    target = next(i for i in items if i.trace.success)
    request = ReuseRequest(target.task.embedding, target.trace.producer, (application,),
                           tuple(s.operator for s in target.subtraces))
    assert store.whole(request)[0].trace_id == target.trace.trace_id
    assert store.cascade(request, theta=0.99).route == "whole"
    assert store.query("subtraces", request).route == "subtraces"
