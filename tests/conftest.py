import os

import pytest

from expreuse.embedding import HashEmbedder
from expreuse.stores import make_store
from expreuse.synthetic import make_traces

DIM = 64


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder(DIM)


@pytest.fixture(scope="session")
def traces(embedder):
    return (make_traces(150, seed=1, producer="llm", embedder=embedder)
            + [t for t in make_traces(150, seed=2, producer="policy", embedder=embedder)])


def _open(kind):
    """Realizations needing a service are skipped unless it is configured:
    EXPREUSE_TEST_DSN for PostgreSQL; Qdrant runs in-process."""
    if kind in ("relational", "jsonb"):
        dsn = os.environ.get("EXPREUSE_TEST_DSN")
        if not dsn:
            pytest.skip("EXPREUSE_TEST_DSN not set")
        pytest.importorskip("psycopg")
        return make_store(kind, DIM, dsn=dsn, schema=f"expreuse_test_{kind}")
    if kind == "qdrant":
        pytest.importorskip("qdrant_client")
    return make_store(kind, DIM)


@pytest.fixture(scope="session", params=["memory", "relational", "jsonb", "qdrant"])
def store(request, traces):
    s = _open(request.param)
    s.setup(reset=True)
    for item in traces:
        s.insert(item)
    yield s
    s.close()


@pytest.fixture(scope="session")
def oracle(traces):
    s = make_store("memory", DIM)
    for item in traces:
        s.insert(item)
    return s
