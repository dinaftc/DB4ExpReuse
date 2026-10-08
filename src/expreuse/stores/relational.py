"""Normalized relational realization in PostgreSQL with pgvector."""
from __future__ import annotations

from collections import defaultdict

from ..model import ReuseRequest, Step, StoredTrace, Subtrace, SubtraceHit, TraceHit
from ._pg import RANKED_GOALS, PostgresStore

WHOLE = f"""
SELECT tr.trace_id, tr.task_id, g.application, tr.producer, g.goal,
       1 - g.distance AS similarity
FROM (
    SELECT r.* FROM ({RANKED_GOALS}) r
    WHERE EXISTS (SELECT 1 FROM task t JOIN trace tr USING (task_id)
                  WHERE t.goal_id = r.goal_id AND tr.producer = %(producer)s
                    AND tr.success)
    ORDER BY r.distance FETCH FIRST %(k)s ROWS WITH TIES
) g
JOIN task t USING (goal_id)
JOIN trace tr USING (task_id)
WHERE tr.producer = %(producer)s AND tr.success
ORDER BY g.distance, tr.length, tr.trace_id
LIMIT %(k)s
"""

SUBTRACES = f"""
SELECT c.* FROM unnest(%(operators)s::text[]) AS o(operator)
CROSS JOIN LATERAL (
    SELECT s.trace_id, s.subgoal_id, g.application, tr.producer, g.goal,
           1 - g.distance AS similarity, s.operator, s.binding
    FROM (
        SELECT r.* FROM ({RANKED_GOALS}) r
        WHERE EXISTS (SELECT 1 FROM task t JOIN trace tr USING (task_id)
                      JOIN subtrace s USING (trace_id)
                      WHERE t.goal_id = r.goal_id AND tr.producer = %(producer)s
                        AND s.operator = o.operator AND s.finished AND s.fits)
        ORDER BY r.distance FETCH FIRST %(k)s ROWS WITH TIES
    ) g
    JOIN task t USING (goal_id)
    JOIN trace tr USING (task_id)
    JOIN subtrace s USING (trace_id)
    WHERE tr.producer = %(producer)s AND s.operator = o.operator
      AND s.finished AND s.fits
    ORDER BY g.distance, s.length, s.trace_id, s.subgoal_id
    LIMIT %(k)s
) c
"""


class RelationalStore(PostgresStore):
    schema_file = "relational.sql"
    default_schema = "expreuse_rel"

    def insert(self, item: StoredTrace) -> None:
        lengths = defaultdict(int)
        for s in item.steps:
            lengths[s.subgoal_id] += 1
        from psycopg.types.json import Jsonb

        with self.conn.transaction(), self.conn.cursor() as cur:
            goal_id = self._upsert_goal(cur, item)
            cur.execute("INSERT INTO task VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (item.task.task_id, item.task.application, goal_id))
            cur.execute("INSERT INTO trace VALUES (%s, %s, %s, %s, %s)",
                        (item.trace.trace_id, item.trace.task_id, item.trace.producer,
                         item.trace.success, item.length))
            cur.executemany(
                "INSERT INTO subtrace VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(s.trace_id, s.subgoal_id, s.operator, Jsonb(s.binding), s.finished,
                  s.fits, lengths[s.subgoal_id]) for s in item.subtraces])
            cur.executemany(
                "INSERT INTO step VALUES (%s, %s, %s, %s, %s)",
                [(s.trace_id, s.step_index, s.subgoal_id, s.action, s.observation)
                 for s in item.steps])

    def _steps(self, cur, trace_ids) -> dict[str, list[Step]]:
        steps = defaultdict(list)
        if trace_ids:
            cur.execute(
                """SELECT trace_id, step_index, subgoal_id, action, observation
                   FROM step WHERE trace_id = ANY(%s::text[])
                   ORDER BY trace_id, step_index""", (list(trace_ids),))
            for row in cur:
                steps[row[0]].append(Step(*row))
        return steps

    def whole(self, request: ReuseRequest) -> list[TraceHit]:
        with self.conn.cursor() as cur:
            rows = cur.execute(WHOLE, self._params(request)).fetchall()
            ids = [r[0] for r in rows]
            steps = self._steps(cur, ids)
            subs = defaultdict(list)
            if ids:
                cur.execute(
                    """SELECT trace_id, subgoal_id, operator, binding, finished, fits
                       FROM subtrace WHERE trace_id = ANY(%s::text[])
                       ORDER BY trace_id, subgoal_id""", (ids,))
                for row in cur:
                    subs[row[0]].append(Subtrace(*row))
        self.conn.commit()
        return [TraceHit(*r[:5], float(r[5]), steps[r[0]], subs[r[0]]) for r in rows]

    def subtraces_for(self, request, operators):
        out = {op: [] for op in operators}
        if not operators:
            return out
        with self.conn.cursor() as cur:
            rows = cur.execute(
                SUBTRACES, {**self._params(request), "operators": operators}).fetchall()
            steps = self._steps(cur, {r[0] for r in rows})
        self.conn.commit()
        for r in rows:
            own = [s for s in steps[r[0]] if s.subgoal_id == r[1]]
            out[r[6]].append(SubtraceHit(*r[:5], float(r[5]), r[6], r[7], own))
        return out

    def counts(self) -> dict[str, int]:
        with self.conn.cursor() as cur:
            out = {name: cur.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                   for name, table in [("tasks", "task"), ("traces", "trace"),
                                       ("subtraces", "subtrace"), ("steps", "step"),
                                       ("goals", "goal")]}
        self.conn.commit()
        return out
