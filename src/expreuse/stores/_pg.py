"""Shared PostgreSQL plumbing for the relational and JSONB realizations."""
from __future__ import annotations

from importlib import resources

import numpy as np

from ..embedding import normalize
from .base import ExperienceStore


def vector_literal(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.9g}" for x in normalize(embedding).tolist()) + "]"


def ensure_pgvector(conn) -> None:
    # Skip CREATE EXTENSION when pgvector is already there: it may have been
    # installed by an administrator for a role that cannot create it.
    if not conn.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'").fetchone():
        conn.execute("CREATE EXTENSION vector")


# Goals of the requested applications, nearest first. Kept as an ordered
# subquery so the eligibility filter applied on top of it is evaluated lazily,
# stopping after the K nearest eligible goals (plus ties).
RANKED_GOALS = """
    SELECT goal_id, application, goal, embedding <=> %(q)s::vector AS distance
    FROM goal WHERE application = ANY(%(applications)s::text[])
    ORDER BY distance
"""


class PostgresStore(ExperienceStore):
    schema_file: str
    default_schema: str

    def __init__(self, dsn: str, dim: int, schema: str | None = None):
        import psycopg

        self.dim = dim
        self.schema = schema or self.default_schema
        self.conn = psycopg.connect(dsn)
        ensure_pgvector(self.conn)
        self.conn.commit()
        self._use_schema()

    def _use_schema(self) -> None:
        self.conn.execute(f'SET search_path TO "{self.schema}", public')
        self.conn.commit()

    def setup(self, reset: bool = False) -> None:
        if reset:
            self.conn.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
        self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        self._use_schema()
        ddl = resources.files("expreuse.stores").joinpath(self.schema_file).read_text()
        self.conn.execute(ddl.replace("{dim}", str(self.dim)))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _upsert_goal(self, cur, item) -> int:
        if item.task.embedding is None:
            raise ValueError("task has no goal embedding")
        cur.execute(
            """INSERT INTO goal (application, goal, embedding)
               VALUES (%s, %s, %s::vector)
               ON CONFLICT (application, goal) DO UPDATE SET goal = EXCLUDED.goal
               RETURNING goal_id""",
            (item.task.application, item.task.goal, vector_literal(item.task.embedding)))
        return cur.fetchone()[0]

    def _params(self, request) -> dict:
        return {"q": vector_literal(request.embedding),
                "applications": list(request.applications),
                "producer": request.producer, "k": request.k}

    def create_goal_index(self, m: int = 16, ef_construction: int = 64) -> None:
        """Optional HNSW index for approximate goal search (Section 4.5)."""
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS goal_hnsw_idx ON goal USING hnsw "
            f"(embedding vector_cosine_ops) WITH (m = {int(m)}, "
            f"ef_construction = {int(ef_construction)})")
        self.conn.commit()

    def drop_goal_index(self) -> None:
        self.conn.execute("DROP INDEX IF EXISTS goal_hnsw_idx")
        self.conn.commit()
