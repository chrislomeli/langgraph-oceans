from __future__ import annotations

import logging

from langchain_core.documents import Document
from stores.postgres import PgGateway, get_pg_gateway
from stores.postgres.embedder import Embedder

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Fragment stores backed entirely by Postgres + pgvector."""

    def __init__(self, pg_gateway: PgGateway | None = None, embedder: Embedder | None = None):
        self._pg = pg_gateway or get_pg_gateway()
        self._embedder = embedder or Embedder()

    def search(
            self,
            query: str,
            k: int = 5,
            species: str | None = None,
            doc_type: str | None = None,
    ) -> list[dict]:
        """Vector top-k with an optional metadata pre-filter.

        species/doc_type narrow the candidate set BEFORE ranking (the scoping that
        stops a humpback query from returning blue-whale chunks); the embedding then
        ranks what's left by cosine distance.
        """
        qvec = "[" + ",".join(map(str, self._embedder.embed(query))) + "]"

        where, params = [], []
        if species:
            where.append("%s = any(species)")
            params.append(species)
        if doc_type:
            where.append("doc_type = %s")
            params.append(doc_type)
        where_sql = ("where " + " and ".join(where)) if where else ""

        sql = f"""
            select chunk_id, species, "section", source, "year",
                   left(text, 100) as snippet,
                   embedding <=> %s::vector as distance
            from doc_chunks
            {where_sql}
            order by distance
            limit %s
        """
        return self._pg.fetch_rows(sql, (qvec, *params, k))

    def search_hybrid(
            self,
            query: str,
            k: int = 5,
            species: str | None = None,
            doc_type: str | None = None,
            cand: int = 20,
            rrf_k: int = 60,
    ) -> list[dict]:
        """Hybrid retrieval: fuse vector (semantic) + tsv (keyword) via RRF.

        Each channel produces a top-`cand` ranked list over the SAME pre-filtered
        candidates; a chunk's fused score = 1/(rrf_k+vrank) + 1/(rrf_k+krank), so a
        chunk strong in EITHER channel surfaces. Rank-based fusion sidesteps the fact
        that cosine distance and ts_rank live on incomparable scales.
        """
        qvec = "[" + ",".join(map(str, self._embedder.embed(query))) + "]"

        where, fparams = [], []
        if species:
            where.append("%s = any(species)")
            fparams.append(species)
        if doc_type:
            where.append("doc_type = %s")
            fparams.append(doc_type)
        where_sql = ("where " + " and ".join(where)) if where else ""

        sql = f"""
            with filtered as (
                select chunk_id, species, "section", source, "year", text, embedding, tsv
                from doc_chunks
                {where_sql}
            ),
            vec as (
                select chunk_id, row_number() over (order by embedding <=> %s::vector) as rank
                from filtered
                order by rank
                limit %s
            ),
            kw as (
                select chunk_id, row_number() over (order by ts_rank(tsv, q) desc) as rank
                from filtered, websearch_to_tsquery('english', %s) q
                where tsv @@ q
                order by rank
                limit %s
            ),
            fused as (
                select chunk_id, v.rank as vrank, k.rank as krank,
                       coalesce(1.0 / (%s + v.rank), 0) + coalesce(1.0 / (%s + k.rank), 0) as score
                from vec v
                full outer join kw k using (chunk_id)
            )
            select d.chunk_id, d.species, d."section", d.source, d."year",
                   left(d.text, 100) as snippet, f.vrank, f.krank, f.score
            from fused f
            join doc_chunks d using (chunk_id)
            order by f.score desc
            limit %s
        """
        params = (*fparams, qvec, cand, query, cand, rrf_k, rrf_k, k)
        return self._pg.fetch_rows(sql, params)

    def save_chunks(self, chunks: list[Document]) -> None:
        """Embed all chunks in one batch pass, then upsert to Postgres."""
        if not chunks:
            return
        # embed header + body so the vector carries species+topic; the bare body
        # (page_content) is what lands in the text column for citation.
        texts = [
            f"{f.metadata['header']}\n{f.page_content}"
            for f in chunks
        ]
        embeddings = self._embedder.embed_batch(texts)
        self.upsert_doc_chunks(documents=chunks, embeddings=embeddings)

    def upsert_doc_chunks(
            self,
            documents: list[Document],
            embeddings: list[list[float]] | None = None,
    ) -> None:

        records = []
        for doc, vec in zip(documents, embeddings):
            metadata = doc.metadata
            records.append([
                metadata["chunk_id"],  # chunk_id
                doc.page_content,  # text
                metadata["header"],  # heading
                [metadata["species"]],  # species
                metadata.get("sanctuary"),  # sanctuary (None for SAR, not '')
                metadata["source_tag"],  # doc_type
                metadata["section"],  # section
                metadata["stock"],  # stock
                metadata["file"],  # source
                metadata["year"],  # year
                vec,  # embedding
                self._embedder.version  # embedder_ver
            ])

        sql = """
              insert into doc_chunks (chunk_id, text, header, species, sanctuary, doc_type, "section", stock, source, 
                                      "year", embedding, embedder_ver)
              values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
              on conflict (chunk_id) do update set
                  text         = excluded.text,
                  header       = excluded.header,
                  species      = excluded.species,
                  sanctuary    = excluded.sanctuary,
                  doc_type     = excluded.doc_type,
                  "section"    = excluded."section",
                  stock        = excluded.stock,
                  source       = excluded.source,
                  "year"       = excluded."year",
                  embedding    = excluded.embedding,
                  embedder_ver = excluded.embedder_ver
              """
        self._pg.execute_many(sql, records)
