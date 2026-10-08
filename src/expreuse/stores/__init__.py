from .base import ExperienceStore
from .memory import MemoryStore

REALIZATIONS = ("memory", "relational", "jsonb", "qdrant")


def make_store(kind: str, dim: int, dsn: str | None = None, url: str | None = None,
               **options) -> ExperienceStore:
    """Open a physical realization. `dsn` is the PostgreSQL connection string
    (relational, jsonb); `url` is the Qdrant endpoint (in-process if omitted)."""
    if kind == "memory":
        return MemoryStore()
    if kind == "relational":
        from .relational import RelationalStore
        return RelationalStore(dsn, dim, **options)
    if kind == "jsonb":
        from .jsonb import JsonbStore
        return JsonbStore(dsn, dim, **options)
    if kind == "qdrant":
        from .qdrant import QdrantStore
        return QdrantStore(dim, url=url, **options)
    raise ValueError(f"unknown realization: {kind}")
