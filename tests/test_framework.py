import numpy as np

from expreuse import PlanStep, ReuseConfig, Task, run_task
from expreuse.consumers import LLMConsumer, ShapedPolicy, render_experience
from expreuse.evaluation import success_rate
from expreuse.grounding import ground, substitute
from expreuse.io import from_json, to_json
from expreuse.model import Experience, Step, Subtrace, SubtraceHit, TraceHit
from expreuse.operators import ALFWORLD, SCIENCEWORLD, canonical
from expreuse.stores import MemoryStore


def test_operators():
    assert canonical("Pick  Up") == "take" and canonical("move") == "put"
    assert ALFWORLD.shared(SCIENCEWORLD) == {"take", "put", "heat", "cool"}


def test_substitute_is_simultaneous_and_whole_word():
    mapping = {"mug 2": "mug 4", "mug 4": "mug 1", "microwave 1": "microwave 2"}
    assert substitute("heat mug 2 with microwave 1, not mug 4 or mug 21", mapping) == \
        "heat mug 4 with microwave 2, not mug 1 or mug 21"


def _hit(binding, actions, operator="heat", subgoal_id=1):
    steps = [Step("t", i, subgoal_id, a, "ok") for i, a in enumerate(actions)]
    return SubtraceHit("t", subgoal_id, "alfworld", "p", "g", 0.9, operator, binding, steps)


def test_ground_subtraces():
    plan = (PlanStep("take", {"object": "mug 4"}),
            PlanStep("heat", {"object": "mug 4", "appliance": "microwave 2"}))
    hit = _hit({"object": "mug 2", "appliance": "microwave 1"},
               ["go to microwave 1", "heat mug 2 with microwave 1"])
    out = ground(Experience("subtraces", subtraces=[[], [hit]]), plan)
    assert [s.action for s in out.subtraces[1][0].steps] == [
        "go to microwave 2", "heat mug 4 with microwave 2"]
    assert out.subtraces[1][0].binding == {"object": "mug 4", "appliance": "microwave 2"}


def test_ground_whole_aligns_by_operator():
    plan = (PlanStep("take", {"object": "cup 1", "source": "shelf 3"}),
            PlanStep("put", {"object": "cup 1", "destination": "cabinet 2"}))
    steps = [Step("t", 0, 0, "take mug 2 from countertop 1", "ok"),
             Step("t", 1, 1, "put mug 2 in/on fridge 1", "ok")]
    subs = [Subtrace("t", 0, "take", {"object": "mug 2", "source": "countertop 1"}, True, True),
            Subtrace("t", 1, "put", {"object": "mug 2", "destination": "fridge 1"}, True, True)]
    trace = TraceHit("t", "task", "alfworld", "p", "g", 0.9, steps, subs)
    out = ground(Experience("whole", traces=[trace]), plan)
    assert [s.action for s in out.traces[0].steps] == [
        "take cup 1 from shelf 3", "put cup 1 in/on cabinet 2"]


class UniformPolicy:
    def log_probs(self, task, history, admissible):
        return [0.0] * len(admissible)


def test_shaped_policy_follows_experience(embedder):
    task = Task("x", "alfworld", "put a hot mug in cabinet")
    hit = _hit({}, ["go to microwave 1", "heat mug 2 with microwave 1"])
    hit.goal = task.goal
    policy = ShapedPolicy(UniformPolicy(), embedder, m=1, lam=1.0)
    policy.begin(task, (), Experience("subtraces", subtraces=[[hit]]))
    admissible = ["look", "heat mug 2 with microwave 1", "go to microwave 1"]
    assert policy.act([("", "start")], admissible) == "go to microwave 1"
    q = policy.q_itd([("", "start"), ("go to microwave 1", "ok")], admissible)
    assert np.allclose(q, [0, 1, 0])


def test_shaped_policy_cross_application_matches_operators(embedder):
    task = Task("x", "scienceworld", "boil water")
    hit = _hit({}, ["heat mug 2 with microwave 1"])
    hit.goal = "boil water in a mug"
    operator_of = lambda a: "heat" if a.startswith("activate stove") else None
    policy = ShapedPolicy(UniformPolicy(), embedder, operator_of=operator_of)
    policy.begin(task, (), Experience("subtraces", subtraces=[[hit]]),
                 cross_application=True)
    admissible = ["look around", "activate stove", "heat mug 2 with microwave 1"]
    assert policy.act([("", "start")], admissible) == "activate stove"


class Echo:
    def __init__(self):
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return "> take mug 1 from shelf 1\nbecause"


def test_llm_consumer_prompt():
    model = Echo()
    consumer = LLMConsumer(model, "You act in a house.", with_plan=True)
    hit = _hit({}, ["heat mug 2 with microwave 1"])
    experience = Experience("subtraces", subtraces=[[hit]])
    consumer.begin(Task("x", "alfworld", "heat a mug"), (PlanStep("heat"),), experience,
                   cross_application=True)
    assert consumer.act([("", "You are in a kitchen.")], []) == "take mug 1 from shelf 1"
    prompt = model.messages[1]["content"]
    assert "Plan: heat" in prompt and render_experience(experience) in prompt
    assert "different application" in prompt and "You are in a kitchen." in prompt


def test_bootstrap():
    mean, se = success_rate([[1, 0, 1, 1], [1, 0, 0, 1]], resamples=2000)
    assert mean == 62.5 and 15 < se < 30


class ToyLabeler:
    def operator_of(self, action):
        return action.split()[0] if action.split()[0] in ("take", "put") else None

    def completes(self, operator, binding, action, observation):
        return action.startswith(operator) and "Nothing" not in observation

    def binding(self, operator, steps):
        return {"object": steps[-1][0].split()[1]}


class ToyApp:
    name = "toy"
    labeler = ToyLabeler()

    def plan(self, task):
        return (PlanStep("take", {"object": "key"}), PlanStep("put", {"object": "key"}))

    def reset(self, task):
        self.done = []
        return "A key lies here."

    def step(self, action):
        self.done.append(action)
        success = self.done[-2:] == ["take key", "put key"]
        return "OK.", success, success

    def admissible_actions(self):
        return ["look", "take key", "put key"]

    def max_steps(self, task):
        return 6


class Scripted:
    def __init__(self, actions):
        self.actions = list(actions)

    def begin(self, task, plan, experience, cross_application=False):
        self.experience = experience

    def act(self, history, admissible):
        return self.actions.pop(0)


def test_run_task_stores_then_reuses(embedder):
    store, app = MemoryStore(), ToyApp()
    first = run_task(app, Scripted(["look", "take key", "put key"]),
                     Task("t1", "toy", "store the key"), embedder, store,
                     store_as="scripted")
    assert first.success and store.counts()["subtraces"] == 2
    assert [s.subgoal_id for s in first.stored.steps] == [0, 0, 1]
    assert all(s.finished and s.fits for s in first.stored.subtraces)

    consumer = Scripted(["take key", "put key"])
    config = ReuseConfig("cascade", producer="scripted", theta=0.6)
    second = run_task(app, consumer, Task("t2", "toy", "store the key"), embedder,
                      store, config)
    assert second.success and consumer.experience.route == "whole"
    assert store.counts()["traces"] == 1

    item = first.stored
    assert to_json(from_json(to_json(item))) == to_json(item)


def test_expert_consumer_follows_the_application_expert(embedder):
    from expreuse.consumers import ExpertConsumer

    class ExpertApp(ToyApp):
        def expert_action(self):
            return ["take key", "put key"][len(self.done)] if len(self.done) < 2 else None

    app = ExpertApp()
    episode = run_task(app, ExpertConsumer(app), Task("t", "toy", "store the key"), embedder,
                       MemoryStore(), store_as="expert")
    assert episode.success and [s.action for s in episode.stored.steps] == ["take key", "put key"]
