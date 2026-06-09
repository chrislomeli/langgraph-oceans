"""evals.photo_id — Layer-A retrieval eval for the photo_id tool (roadmap item 6).

The first concrete eval on top of `evals.framework`. Establishes a FIXED baseline
of closed-set leave-one-out re-identification, scored Precision/Recall@k + MRR, so
the Phase-C LoRA lift is measurable against the same yardstick. The held-out split
is frozen in `datasets.photo_id_eval_split` (see build_split.sql) — not recomputed
at run time. Scoring itself is live: each held-out vector is searched against the
real catalog index (the thing under measurement).
"""
