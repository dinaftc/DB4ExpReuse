"""The two applications without their simulators: plans from goals, step labelling, segmentation of the sample
traces, and the reuse queries over them."""
import csv
import json
from pathlib import Path

import pytest

from expreuse import ReuseRequest
from expreuse.embedding import HashEmbedder
from expreuse.envs.alfworld import AlfworldLabeler, bind_plan, goal_entities, goal_operators
from expreuse.envs.scienceworld import ScienceWorldLabeler, plan_operators, task_type
from expreuse.io import read_jsonl
from expreuse.model import PlanStep
from expreuse.operators import ALFWORLD, SCIENCEWORLD
from expreuse.segmentation import segment
from expreuse.stores import MemoryStore

DATA = Path(__file__).resolve().parents[1] / "data"
LABELERS = {"alfworld": AlfworldLabeler, "scienceworld": ScienceWorldLabeler}
LIBRARIES = {"alfworld": ALFWORLD, "scienceworld": SCIENCEWORLD}


def test_alfworld_plan_from_goal():
    assert goal_operators("put a hot cup in cabinet") == ("take", "heat", "put")
    assert goal_operators("put a cool mug in cabinet") == ("take", "cool", "put")
    assert goal_operators("clean some soapbar and put it in garbagecan.") == ("take", "clean", "put")
    assert goal_operators("examine the book with the desklamp") == ("take", "use")
    assert goal_operators("put two creditcard in desk.") == ("take", "put", "take", "put")
    assert goal_entities("put a hot cup in cabinet") == ("cup", "cabinet")


def test_alfworld_plan_is_bound_from_the_scan():
    room = {"countertop 1": ["cup 2", "apple 1"], "microwave 1": [], "cabinet 3": []}
    plan = bind_plan(("take", "heat", "put"), "put a hot cup in cabinet", room)
    assert plan == (PlanStep("take", {"object": "cup 2", "source": "countertop 1"}),
                    PlanStep("heat", {"object": "cup 2", "appliance": "microwave 1"}),
                    PlanStep("put", {"object": "cup 2", "destination": "cabinet 3"}))


def test_alfworld_labeler():
    labeler = AlfworldLabeler("put a hot cup in cabinet")
    actions = ["go to countertop 1", "take cup 1 from countertop 1", "put cup 1 in/on microwave 1",
               "heat cup 1 with microwave 1", "take cup 1 from microwave 1", "put cup 1 in/on cabinet 3"]
    assert [labeler.operator_of(a) for a in actions] == [None, "take", None, "heat", None, "put"]
    assert labeler.completes("heat", {}, "heat cup 1 with microwave 1", "You heat the cup 1 using the microwave 1.")
    assert not labeler.completes("heat", {}, "heat cup 1 with microwave 1", "Nothing happens.")
    assert labeler.binding("take", [("go to countertop 1", "ok"), ("take cup 1 from countertop 1", "ok")]) == \
        {"object": "cup 1", "source": "countertop 1"}


def test_scienceworld_plan_from_description():
    paint = "Your task is to use chemistry to create violet paint. When you are done, focus on the violet paint."
    assert plan_operators(paint) == ("pour", "mix", "focus on")
    animal = "Your task is to find a(n) animal. First, focus on the thing. Then, move it to the orange box in the kitchen."
    assert task_type(animal) == "find-animal" and plan_operators(animal) == ("focus on", "take", "put")
    assert plan_operators("", "melt")[-1] == "measure"


def test_scienceworld_labeler():
    labeler = ScienceWorldLabeler()
    actions = ["teleport to kitchen", "pick up thermometer", "move lead to blast furnace", "activate blast furnace",
               "use thermometer on lead", "pour yellow paint into cup", "mix cup", "focus on violet paint", "look around"]
    assert [labeler.operator_of(a) for a in actions] == \
        [None, "take", "put", "activate", "measure", "pour", "mix", "focus on", None]
    assert labeler.completes("take", {}, "pick up thermometer", "You move the thermometer to the inventory.")
    assert not labeler.completes("take", {}, "pick up thermometer", "No known action matches that input.")
    assert labeler.binding("take", [("teleport to kitchen", "ok"), ("pick up thermometer", "ok")]) == \
        {"object": "thermometer", "source": "kitchen"}


@pytest.mark.parametrize("application", ["alfworld", "scienceworld"])
def test_sample_operators_have_signatures(application):
    operators = json.loads((DATA / application / "operators.json").read_text())
    for row in operators:
        assert row["operator"] in LIBRARIES[application]
    stored = {s["operator"] for s in csv.DictReader(open(DATA / application / "subtraces.csv"))}
    assert stored <= {row["operator"] for row in operators}


@pytest.mark.parametrize("application", ["alfworld", "scienceworld"])
def test_sample_traces_segment_with_the_labeler(application):
    """Re-segmenting a sample trace with the application's labeler and its stored plan reproduces its stored
    steps and subtraces."""
    plans = {}
    for row in csv.DictReader(open(DATA / application / "plans.csv")):
        plans.setdefault(row["task_id"], []).append(PlanStep(row["operator"], json.loads(row["binding"])))
    for item in read_jsonl(DATA / application / "traces.jsonl"):
        labeler = LABELERS[application](item.task.goal) if application == "alfworld" else LABELERS[application]()
        raw = [(s.action, s.observation) for s in item.steps]
        steps, subtraces = segment(item.trace.trace_id, tuple(plans[item.task.task_id]), raw, labeler)
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
