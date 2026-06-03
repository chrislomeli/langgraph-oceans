"""Postgres-backed store layer.

Seed ships the connection gateway only. Domain repositories
(e.g. ``conservation_repo``) are added per project alongside it.
"""

from stores.postgres.gateway import PgGateway, get_pg_gateway

__all__ = ["PgGateway", "get_pg_gateway"]
