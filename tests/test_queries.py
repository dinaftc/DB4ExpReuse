import pytest

from expreuse.model import ReuseRequest
from expreuse.synthetic import make_traces


@pytest.fixture(scope="module")
def requests(embedder):
    # Goals drawn from the same generator with another seed: some match stored
    # goals exactly, others only partially.
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
