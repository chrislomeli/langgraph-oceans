"""pg_gateway.py — Postgres + pgvector access layer.

Provides PgGateway, a psycopg3 connection-pool wrapper with two tiers of helpers:

    Tier 1 — raw SQL primitives:
        fetch_rows()        — SELECT → list[dict]
        execute()           — INSERT / UPDATE / DELETE → rowcount

    Tier 2 — entity upserts (used by the write-through wrappers):
        ensure_session()    — sessions (idempotent)
        upsert_exchanges()  — exchanges (bulk)
        upsert_thread()     — threads + thread_exchanges junction
        upsert_fragment()   — fragments + fragment_exchanges junction (embedding optional)
        upsert_profile()    — user_profiles (single row)

    Plus the pgvector search path:
        search_similar()    — cosine top-k on fragments.embedding

Dependencies (add to pyproject.toml):
    psycopg[binary]>=3.2
    psycopg-pool>=3.2
    pgvector>=0.3
    numpy>=1.26

Schema lives in raw/schema.sql — run once:
    psql $POSTGRES_URL -f raw/schema.sql
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from config import get_settings

logger = logging.getLogger(__name__)

# Must match raw/schema.sql vector(N). 384 = sentence-transformers/all-MiniLM-L6-v2
# (the fastembed default used by Embedder). Change both if you swap models.
EMBEDDING_DIM = 384


class PgGateway:
    """Postgres + pgvector access layer backed by a connection pool.

    Create once at app startup and share across graph nodes.
    """

    def __init__(self, min_size: int = 2, max_size: int = 10):
        url = get_settings().postgres_url
        # open=False defers actual connections until first use so import-time
        # startup doesn't fail if Postgres isn't running.
        self._pool = ConnectionPool(
            conninfo=url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=False,
            configure=register_vector,
        )

    def open(self) -> None:
        """Open the pool. Call once at app startup."""
        self._pool.open(wait=True)
        logger.info("PgGateway pool open (min=%d max=%d)", self._pool.min_size, self._pool.max_size)

    def close(self) -> None:
        """Drain the pool. Call at app shutdown."""
        self._pool.close()

    @contextmanager
    def conn(self) -> Generator:
        """Borrow a connection from the pool; psycopg3 auto-commits on clean exit."""
        with self._pool.connection() as conn:
            yield conn

    # ══════════════════════════════════════════════════════════════════════════
    # Tier 1 — raw SQL primitives
    # ══════════════════════════════════════════════════════════════════════════

    def _format_sql_for_log(self, sql: str, params: tuple) -> str:
        """Format SQL and params for debug logging. Truncates long statements."""
        max_len = 500
        sql_preview = sql if len(sql) <= max_len else sql[:max_len] + "... [truncated]"
        params_str = (
            str(params) if len(str(params)) <= 200 else str(params)[:200] + "... [truncated]"
        )
        return f"SQL: {sql_preview} | Params: {params_str}"

    def fetch_rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Run a SELECT; return every row as a dict."""
        try:
            with self.conn() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except Exception as e:
            logger.exception(
                "fetch_rows failed: %s | %s",
                e,
                self._format_sql_for_log(sql, params),
            )
            raise

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Run a mutating statement; return rowcount."""
        try:
            with self.conn() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount
        except Exception as e:
            logger.exception(
                "execute failed: %s | %s",
                e,
                self._format_sql_for_log(sql, params),
            )
            raise


# ── Module-level singleton (lazy) ──────────────────────────────────────────────

_gateway: PgGateway | None = None


def get_pg_gateway() -> PgGateway:
    """Return the module-level PgGateway singleton, opening the pool on first call."""
    global _gateway
    if _gateway is None:
        _gateway = PgGateway()
        _gateway.open()
    return _gateway
