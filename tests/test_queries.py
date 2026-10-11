import numpy as np
import pytest

from expreuse.model import ReuseRequest, Step, StoredTrace, Subtrace, Task, Trace
from expreuse.stores import MemoryStore
from expreuse.synthetic import make_traces

from conftest import DIM, open_realization


@pytest.fixture(scope="module")
def requests(embedder):
    targets = make_traces(40, seed=99, embedder=embedder)
    return [ReuseRequest(t.task.embedding, producer, ("alfworld",),
                         tuple(s.operator for s in t.subtraces) or ("take",), k)
            for t in targets for producer in ("llm", "policy") for k in (1, 3)]


def trace_key(h):
    return (h.trace_id, [(s.step_index, s.subgoal_id, s.action, s.observation)
                         for s in h.steps],
            [(s.subgoal_id, s.operator, s.binding, s.finished, s.fits)
             for s in h.subtraces])


def sub_key(h):
    return (h.trace_id, h.subgoal_id, h.operator, h.binding,
            [(s.step_index, s.action, s.observation) for s in h.steps])


def test_whole_definition(store, requests, traces):
    by_id = {t.trace.trace_id: t for t in traces}
    for r in requests:
        hits = store.whole(r)
        assert len(hits) == r.k
        for h in hits:
            src = by_id[h.trace_id]
            assert src.trace.success and src.trace.producer == r.producer
            assert h.goal == src.task.goal and h.length == src.length
        assert [h.similarity for h in hits] == sorted((h.similarity for h in hits),
                                                      reverse=True)


def test_subtraces_definition(store, requests, traces):
    subs = {(s.trace_id, s.subgoal_id): (s, t) for t in traces for s in t.subtraces}
    for r in requests:
        answer = store.subtraces(r)
        assert len(answer) == len(r.plan)
        for operator, hits in zip(r.plan, answer):
            for h in hits:
                sub, src = subs[h.trace_id, h.subgoal_id]
                assert sub.operator == operator and sub.finished and sub.fits
                assert src.trace.producer == r.producer
                assert [s.action for s in h.steps] == [
                    s.action for s in src.subtrace_steps(sub.subgoal_id)]


def test_realizations_agree(store, oracle, requests):
    for r in requests:
        assert [trace_key(h) for h in store.whole(r)] == [
            trace_key(h) for h in oracle.whole(r)]
        assert [[sub_key(h) for h in hits] for hits in store.subtraces(r)] == [
            [sub_key(h) for h in hits] for hits in oracle.subtraces(r)]
        for theta in (0.3, 0.7, 0.999, 1.5):
            got, want = store.cascade(r, theta), oracle.cascade(r, theta)
            assert got.route == want.route
            assert [h.trace_id for h in got.traces] == [h.trace_id for h in want.traces]
        for h, o in zip(store.whole(r), oracle.whole(r)):
            assert h.similarity == pytest.approx(o.similarity, abs=1e-5)


def test_subtraces_reuse_unsuccessful_parents(oracle, requests, traces):
    failed = {t.trace.trace_id for t in traces if not t.trace.success}
    r = ReuseRequest(requests[0].embedding, "llm", ("alfworld",), ("take",), k=300)
    assert failed & {h.trace_id for h in oracle.subtraces(r)[0]}
    assert not failed & {h.trace_id for h in oracle.whole(r)}


def test_cascade_routes(store, requests):
    r = requests[0]
    assert store.cascade(r, -1.0).route == "whole"
    assert store.cascade(r, 1.5).route == "subtraces"


def test_uncovered_and_empty(store, requests):
    r = requests[0]
    partial = ReuseRequest(r.embedding, r.producer, r.applications, ("take", "mix"))
    answer = store.query("subtraces", partial)
    assert answer.route == "subtraces" and answer.uncovered == [1]
    nothing = ReuseRequest(r.embedding, "nobody", r.applications, ("take",))
    assert store.cascade(nothing, 0.5).route == "none"
    other_app = ReuseRequest(r.embedding, r.producer, ("scienceworld",), ("take",))
    assert not store.query("whole", other_app)


def _vector(*coordinates):
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(coordinates)] = coordinates
    return v / np.linalg.norm(v)


def _stored(trace_id, task, producer, success, segments):
    """segments: (operator, finished, fits, number of steps)."""
    steps, subs, i = [], [], 0
    for position, (operator, finished, fits, n) in enumerate(segments):
        for _ in range(n):
            steps.append(Step(trace_id, i, position, f"{operator} step {i}", f"observation {i}"))
            i += 1
        subs.append(Subtrace(trace_id, position, operator, {"object": f"o{position}"}, finished, fits))
    return StoredTrace(task, Trace(trace_id, task.task_id, producer, success), tuple(steps), tuple(subs))


def _hand_built():
    a = Task("A", "alfworld", "put a hot mug in cabinet", _vector(1, 0))
    b = Task("B", "alfworld", "put a cool mug in cabinet", _vector(0.8, 0.6))
    c = Task("C", "scienceworld", "boil water", _vector(0, 1))
    return [_stored("a_ok_long", a, "gemma", True, [("take", 1, 1, 4), ("heat", 1, 1, 3), ("put", 1, 1, 2)]),
            _stored("a_ok_short", a, "gemma", True, [("take", 1, 1, 2), ("heat", 1, 0, 2), ("put", 1, 1, 2)]),
            _stored("a_fail", a, "gemma", False, [("take", 1, 1, 1), ("heat", 0, 1, 5)]),
            _stored("b_ok", b, "dagger", True, [("take", 1, 1, 3), ("cool", 1, 1, 2), ("put", 1, 1, 1)]),
            _stored("b_fail", b, "gemma", False, [("take", 1, 1, 9), ("cool", 1, 1, 1), ("put", 0, 1, 3)]),
            _stored("c_ok", c, "gemma", True, [("take", 1, 1, 1), ("heat", 1, 1, 1), ("measure", 1, 1, 1)])]


def test_definitions_on_hand_built_traces():
    """The query definitions, case by case."""
    store = MemoryStore()
    store.insert_many(_hand_built())
    q = _vector(1, 0.05)
    ids = lambda hits: [h.trace_id for h in hits]
    assert ids(store.whole(ReuseRequest(q, "gemma", ("alfworld",), (), 1))) == ["a_ok_short"]
    assert ids(store.whole(ReuseRequest(q, "gemma", ("alfworld",), (), 5))) == ["a_ok_short", "a_ok_long"]
    assert ids(store.whole(ReuseRequest(q, "dagger", ("alfworld",), (), 5))) == ["b_ok"]
    assert store.whole(ReuseRequest(q, "nobody", ("alfworld",), (), 1)) == []

    answer = store.subtraces(ReuseRequest(q, "gemma", ("alfworld",), ("take", "heat", "put"), 1))
    assert [[(h.trace_id, h.subgoal_id) for h in hits] for hits in answer] == \
        [[("a_fail", 0)], [("a_ok_long", 1)], [("a_ok_long", 2)]]
    assert [s.action for s in answer[1][0].steps] == ["heat step 4", "heat step 5", "heat step 6"]
    answer = store.subtraces(ReuseRequest(q, "gemma", ("alfworld",), ("take", "cool", "clean"), 1))
    assert [(h.trace_id, h.subgoal_id) for h in answer[1]] == [("b_fail", 1)] and answer[2] == []
    answer = store.subtraces(ReuseRequest(q, "gemma", ("alfworld",), ("take", "put", "take", "put"), 1))
    assert answer[0][0].trace_id == answer[2][0].trace_id and answer[1][0].subgoal_id == answer[3][0].subgoal_id
    answer = store.subtraces(ReuseRequest(q, "gemma", ("scienceworld",), ("take", "heat", "put"), 1))
    assert [len(hits) for hits in answer] == [1, 1, 0] and answer[0][0].trace_id == "c_ok"

    high = store.cascade(ReuseRequest(q, "gemma", ("alfworld",), ("take", "cool", "put"), 1), theta=0.6)
    assert high.route == "whole" and high.traces[0].trace_id == "a_ok_short" and high.similarity > 0.99
    low = store.cascade(ReuseRequest(_vector(0.3, 1), "gemma", ("alfworld",), ("take", "cool", "put"), 1), theta=0.9)
    assert low.route == "subtraces" and low.similarity < 0.9 and low.subtraces[1][0].trace_id == "b_fail"
    assert store.cascade(ReuseRequest(_vector(0.3, 1), "gemma", ("alfworld",), ("mix",), 1), theta=0.9).route == "none"
    assert store.query("whole", ReuseRequest(q, "gemma", ("alfworld",), (), 1)).similarity == high.similarity


@pytest.mark.parametrize("kind", ["memory", "relational", "jsonb", "qdrant"])
def test_bulk_loading_matches_single_inserts(kind, traces, oracle, requests):
    store = open_realization(kind, "_bulk")
    store.setup(reset=True)
    store.insert_many(traces)
    assert {k: store.counts()[k] for k in ("traces", "subtraces")} == \
        {k: oracle.counts()[k] for k in ("traces", "subtraces")}
    for r in requests[:20]:
        assert [h.trace_id for h in store.whole(r)] == [h.trace_id for h in oracle.whole(r)]
        assert [[sub_key(h) for h in hits] for hits in store.subtraces(r)] == \
            [[sub_key(h) for h in hits] for hits in oracle.subtraces(r)]
    store.close()
