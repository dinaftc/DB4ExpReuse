"""Vector realization in Qdrant."""
from __future__ import annotations

import threading
import uuid

from ..embedding import normalize
from ..model import ReuseRequest, Step, StoredTrace, Subtrace, SubtraceHit, TraceHit
from .base import ExperienceStore

_NAMESPACE = uuid.UUID("6f1c2f0e-6c59-4b7e-9d7c-0d8a3f6f2b11")
_SCROLL = 1024


def _id(*parts) -> str:
    return str(uuid.uuid5(_NAMESPACE, "\x1f".join(map(str, parts))))


def _key(producer: str, operator: str) -> str:
    return f"{producer}\x1f{operator}"


class QdrantStore(ExperienceStore):
    def __init__(self, dim: int, url: str | None = None, prefix: str = "expreuse",
                 exact: bool = True, hnsw_ef: int | None = None, client=None):
        from qdrant_client import QdrantClient

        self.dim = dim
        self.client = client or (QdrantClient(url=url) if url else QdrantClient(":memory:"))
        self.goals, self.traces, self.subs = (f"{prefix}_{n}" for n in
                                              ("goals", "traces", "subtraces"))
        self.exact, self.hnsw_ef = exact, hnsw_ef
        # Goal eligibility keys are updated read-modify-write; serialize writers
        # of this process. Use a single writer process, or accept that
        # concurrent writers to the same goal must be retried.
        self._lock = threading.Lock()

    def setup(self, reset: bool = False) -> None:
        from qdrant_client import models as m

        for name, vectors in ((self.goals, m.VectorParams(size=self.dim,
                                                         distance=m.Distance.COSINE)),
                              (self.traces, {}), (self.subs, {})):
            if reset and self.client.collection_exists(name):
                self.client.delete_collection(name)
            if not self.client.collection_exists(name):
                self.client.create_collection(name, vectors_config=vectors)
        indexes = {self.goals: ("application", "whole_producers", "subtrace_keys"),
                   self.traces: ("goal_id", "producer"),
                   self.subs: ("goal_id", "producer", "operator")}
        for name, fields in indexes.items():
            for f in fields:
                self.client.create_payload_index(name, f, m.PayloadSchemaType.KEYWORD)

    def close(self) -> None:
        self.client.close()

    def insert(self, item: StoredTrace) -> None:
        from qdrant_client import models as m

        if item.task.embedding is None:
            raise ValueError("task has no goal embedding")
        task, trace = item.task, item.trace
        goal_id = _id(task.application, task.goal)
        common = {"trace_id": trace.trace_id, "goal_id": goal_id,
                  "producer": trace.producer}
        sub_points, keys = [], set()
        for sub in item.subtraces:
            steps = item.subtrace_steps(sub.subgoal_id)
            eligible = sub.finished and sub.fits
            if eligible:
                keys.add(_key(trace.producer, sub.operator))
            sub_points.append(m.PointStruct(
                id=_id(trace.trace_id, sub.subgoal_id), vector={},
                payload={**common, "subgoal_id": sub.subgoal_id,
                         "operator": sub.operator, "binding": sub.binding,
                         "finished": sub.finished, "fits": sub.fits,
                         "eligible": eligible, "length": len(steps),
                         "steps": [[s.step_index, s.action, s.observation]
                                   for s in steps]}))
        trace_point = m.PointStruct(
            id=_id(trace.trace_id), vector={},
            payload={**common, "task_id": task.task_id, "success": trace.success,
                     "length": item.length,
                     "steps": [[s.step_index, s.subgoal_id, s.action, s.observation]
                               for s in item.steps],
                     "subtraces": [[s.subgoal_id, s.operator, s.binding, s.finished,
                                    s.fits] for s in item.subtraces]})
        with self._lock:
            # Items first, goal keys last: a goal never advertises experience
            # that is not yet retrievable.
            self.client.upsert(self.traces, [trace_point])
            if sub_points:
                self.client.upsert(self.subs, sub_points)
            existing = self.client.retrieve(self.goals, [goal_id], with_payload=True)
            payload = existing[0].payload if existing else {
                "application": task.application, "goal": task.goal,
                "whole_producers": [], "subtrace_keys": []}
            producers = set(payload["whole_producers"])
            if trace.success:
                producers.add(trace.producer)
            payload["whole_producers"] = sorted(producers)
            payload["subtrace_keys"] = sorted(keys | set(payload["subtrace_keys"]))
            self.client.upsert(self.goals, [m.PointStruct(
                id=goal_id, vector=normalize(task.embedding).tolist(), payload=payload)])

    def _goals(self, request: ReuseRequest, field: str, value: str) -> dict:
        from qdrant_client import models as m

        result = self.client.query_points(
            self.goals, query=normalize(request.embedding).tolist(),
            query_filter=m.Filter(must=[
                m.FieldCondition(key="application",
                                 match=m.MatchAny(any=list(request.applications))),
                m.FieldCondition(key=field, match=m.MatchValue(value=value))]),
            search_params=m.SearchParams(exact=self.exact, hnsw_ef=self.hnsw_ef),
            limit=request.k, with_payload=["application", "goal"])
        return {str(p.id): (p.score, p.payload) for p in result.points}

    def _items(self, collection: str, goal_ids, conditions: dict) -> list[dict]:
        from qdrant_client import models as m

        if not goal_ids:
            return []
        must = [m.FieldCondition(key="goal_id", match=m.MatchAny(any=list(goal_ids)))]
        must += [m.FieldCondition(key=k, match=m.MatchValue(value=v))
                 for k, v in conditions.items()]
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection, scroll_filter=m.Filter(must=must), limit=_SCROLL,
                offset=offset, with_payload=True)
            out += [p.payload for p in points]
            if offset is None:
                return out

    def whole(self, request: ReuseRequest) -> list[TraceHit]:
        goals = self._goals(request, "whole_producers", request.producer)
        items = self._items(self.traces, goals, {"producer": request.producer,
                                                 "success": True})
        items.sort(key=lambda p: (-goals[p["goal_id"]][0], p["length"], p["trace_id"]))
        hits = []
        for p in items[:request.k]:
            score, goal = goals[p["goal_id"]]
            tid = p["trace_id"]
            hits.append(TraceHit(
                tid, p["task_id"], goal["application"], p["producer"], goal["goal"],
                float(score), [Step(tid, *s) for s in p["steps"]],
                [Subtrace(tid, *s) for s in p["subtraces"]]))
        return hits

    def subtraces_for(self, request, operators):
        out = {}
        for op in operators:
            goals = self._goals(request, "subtrace_keys", _key(request.producer, op))
            items = self._items(self.subs, goals, {
                "producer": request.producer, "operator": op, "eligible": True})
            items.sort(key=lambda p: (-goals[p["goal_id"]][0], p["length"],
                                      p["trace_id"], p["subgoal_id"]))
            out[op] = []
            for p in items[:request.k]:
                score, goal = goals[p["goal_id"]]
                tid, sid = p["trace_id"], p["subgoal_id"]
                out[op].append(SubtraceHit(
                    tid, sid, goal["application"], p["producer"], goal["goal"],
                    float(score), op, p["binding"],
                    [Step(tid, i, sid, a, o) for i, a, o in p["steps"]]))
        return out

    def counts(self) -> dict[str, int]:
        count = lambda name: self.client.count(name, exact=True).count
        return {"goals": count(self.goals), "traces": count(self.traces),
                "subtraces": count(self.subs)}
