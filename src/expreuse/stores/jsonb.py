"""JSONB realization in PostgreSQL: each trace is one document row."""
from __future__ import annotations

from ..model import (Experience, ReuseRequest, Step, StoredTrace, Subtrace,
                     SubtraceHit, TraceHit)
from ._pg import RANKED_GOALS, PostgresStore

_HAS_WHOLE = """EXISTS (SELECT 1 FROM trace_doc d WHERE d.goal_id = r.goal_id
                        AND d.producer = %(producer)s AND d.success)"""

BEST_WHOLE_SIMILARITY = f"""
SELECT 1 - r.distance FROM ({RANKED_GOALS}) r
WHERE {_HAS_WHOLE} ORDER BY r.distance LIMIT 1
"""

WHOLE = f"""
SELECT d.trace_id, d.task_id, g.application, d.producer, g.goal,
       1 - g.distance AS similarity, d.doc
FROM (
    SELECT r.* FROM ({RANKED_GOALS}) r WHERE {_HAS_WHOLE}
    ORDER BY r.distance FETCH FIRST %(k)s ROWS WITH TIES
) g
JOIN trace_doc d USING (goal_id)
WHERE d.producer = %(producer)s AND d.success
ORDER BY g.distance, d.length, d.trace_id
LIMIT %(k)s
"""

SUBTRACES = f"""
SELECT c.* FROM unnest(%(operators)s::text[]) AS o(operator)
CROSS JOIN LATERAL (
    SELECT d.trace_id, (st->>'subgoal_id')::int AS subgoal_id, g.application,
           d.producer, g.goal, 1 - g.distance AS similarity,
           st->>'operator' AS operator, st->'binding' AS binding,
           st->'steps' AS steps
    FROM (
        SELECT r.* FROM ({RANKED_GOALS}) r
        WHERE EXISTS (
            SELECT 1 FROM trace_doc d
            WHERE d.goal_id = r.goal_id AND d.producer = %(producer)s
              AND o.operator = ANY(d.operators))
        ORDER BY r.distance FETCH FIRST %(k)s ROWS WITH TIES
    ) g
    JOIN trace_doc d USING (goal_id)
    CROSS JOIN LATERAL jsonb_array_elements(d.doc->'subtraces') st
    WHERE d.producer = %(producer)s AND o.operator = ANY(d.operators)
      AND st->>'operator' = o.operator
      AND (st->>'finished')::boolean AND (st->>'fits')::boolean
    ORDER BY g.distance, (st->>'length')::int, d.trace_id, subgoal_id
    LIMIT %(k)s
) c
"""


def _step(s: Step) -> dict:
    return {"step_index": s.step_index, "action": s.action, "observation": s.observation}


def _steps(trace_id, subgoal_id, raw) -> list[Step]:
    return [Step(trace_id, s["step_index"], subgoal_id, s["action"], s["observation"])
            for s in raw]


class JsonbStore(PostgresStore):
    schema_file = "jsonb.sql"
    default_schema = "expreuse_jsonb"

    def insert(self, item: StoredTrace) -> None:
        from psycopg.types.json import Jsonb

        doc = {"subtraces": [], "other_steps": []}
        assigned = set()
        for sub in item.subtraces:
            steps = item.subtrace_steps(sub.subgoal_id)
            assigned.add(sub.subgoal_id)
            doc["subtraces"].append({
                "subgoal_id": sub.subgoal_id, "operator": sub.operator,
                "binding": sub.binding, "finished": sub.finished, "fits": sub.fits,
                "length": len(steps), "steps": [_step(s) for s in steps]})
        doc["other_steps"] = [{**_step(s), "subgoal_id": s.subgoal_id}
                              for s in item.steps if s.subgoal_id not in assigned]
        with self.conn.transaction(), self.conn.cursor() as cur:
            goal_id = self._upsert_goal(cur, item)
            eligible = sorted({s.operator for s in item.subtraces
                               if s.finished and s.fits})
            cur.execute("INSERT INTO trace_doc VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (item.trace.trace_id, item.task.task_id, goal_id,
                         item.trace.producer, item.trace.success, item.length,
                         eligible, Jsonb(doc)))

    def whole(self, request: ReuseRequest) -> list[TraceHit]:
        rows = self.conn.execute(WHOLE, self._params(request)).fetchall()
        self.conn.commit()
        hits = []
        for trace_id, task_id, application, producer, goal, similarity, doc in rows:
            steps = [Step(trace_id, s["step_index"], s["subgoal_id"], s["action"],
                          s["observation"]) for s in doc["other_steps"]]
            subs = []
            for st in doc["subtraces"]:
                steps += _steps(trace_id, st["subgoal_id"], st["steps"])
                subs.append(Subtrace(trace_id, st["subgoal_id"], st["operator"],
                                     st["binding"], st["finished"], st["fits"]))
            steps.sort(key=lambda s: s.step_index)
            hits.append(TraceHit(trace_id, task_id, application, producer, goal,
                                 float(similarity), steps, subs))
        return hits

    def subtraces_for(self, request, operators):
        out = {op: [] for op in operators}
        if not operators:
            return out
        rows = self.conn.execute(
            SUBTRACES, {**self._params(request), "operators": operators}).fetchall()
        self.conn.commit()
        for r in rows:
            out[r[6]].append(SubtraceHit(*r[:5], float(r[5]), r[6], r[7],
                                         _steps(r[0], r[1], r[8])))
        return out

    def cascade(self, request: ReuseRequest, theta: float) -> Experience:
        # Route on the goal similarity alone; materialize the trace documents
        # only once WHOLE is selected.
        row = self.conn.execute(BEST_WHOLE_SIMILARITY, self._params(request)).fetchone()
        self.conn.commit()
        if row is not None and float(row[0]) >= theta:
            return Experience("whole", traces=self.whole(request))
        return self._subtraces_or_nothing(request)

    def counts(self) -> dict[str, int]:
        row = self.conn.execute(
            """SELECT count(DISTINCT task_id), count(*),
                      coalesce(sum(jsonb_array_length(doc->'subtraces')), 0),
                      coalesce(sum(length), 0),
                      (SELECT count(*) FROM goal) FROM trace_doc""").fetchone()
        self.conn.commit()
        return dict(zip(("tasks", "traces", "subtraces", "steps", "goals"), map(int, row)))
