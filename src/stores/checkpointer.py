"""checkpointer.py — Async Postgres checkpointer for LangGraph.

The checkpointer persists the full ``OceanState`` between graph super-steps,
keyed by ``thread_id`` (which we set to the session_id).

Lifecycle:
    The async context manager yields a configured AsyncPostgresSaver and tears
    down its connection pool on exit. The checkpointer creates its own async
    connection — separate from the sync PgGateway used by the data layer —
    because LangGraph's async checkpointer requires async psycopg.

Custom serde:
    LangGraph's default JsonPlusSerializer falls back to msgpack for unknown
    types and warns on each unregistered class — and will block them entirely
    in a future version. OceanState currently stores only LangChain message
    types, which JsonPlusSerializer handles natively, so the allow-list is
    empty. Add ``(module, class_name)`` tuples here if we ever persist custom
    domain Pydantic types on the state.

Usage:
    async with make_postgres_checkpointer(url, setup=True) as checkpointer:
        graph = build_sandbox_graph(deps, saver=checkpointer)
        await graph.ainvoke(state, config={"configurable": {"thread_id": sid}})

The ``setup=True`` flag creates the checkpoint tables (idempotent). It is safe
to pass on every run during development; in production, run setup once at
deploy time and pass ``setup=False`` thereafter.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


# Custom domain types to register with the serde, as ``(module, class_name)``
# tuples. Empty today: OceanState stores only LangChain message types, which
# JsonPlusSerializer handles natively. Populate this if we start persisting our
# own Pydantic types on the state, to keep roundtrips from dropping nested fields.
_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = []


def _make_serde() -> JsonPlusSerializer:
    """JsonPlusSerializer registered with our custom domain types (if any).

    Without ``allowed_msgpack_modules`` the serializer logs deprecation
    warnings on every checkpoint load and (in a future LangGraph release)
    will refuse to deserialize them entirely.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def make_postgres_checkpointer(
    postgres_url: str,
    setup: bool = False,

) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield an AsyncPostgresSaver bound to the configured Postgres URL.

    Args:
        postgres_url: Async psycopg connection string for the checkpoint DB.
        setup: If True, create checkpoint tables on entry. Idempotent —
            safe to pass repeatedly. Disable in hot paths where the cost of
            a no-op DDL probe matters.
    """
    async with AsyncPostgresSaver.from_conn_string(postgres_url, serde=_make_serde()) as checkpointer:
        if setup:
            await checkpointer.setup()
        yield checkpointer
