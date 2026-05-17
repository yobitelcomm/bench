# inferencebench-embeddings

Embeddings retrieval plugin for the InferenceBench Suite.

Phase-2-quality skeleton: produces signed envelopes via deterministic
hash-based rankings, with placeholders for real embedding-model invocation
that future revisions wire to TEI / OpenAI / Cohere.

Suite ID: `embeddings.retrieval`

Bundled benchmarks:

- `embeddings.retrieval.beir-mini` — 5 queries × 20-doc corpus, recall@5.
- `embeddings.retrieval.long-doc` — 3 queries with longer documents, nDCG@10.

The skeleton does NOT actually embed any text. For each query it ranks the
corpus by `sha256(query + doc_id)`, then scores the top-k against the
fixture's relevant set. This produces a real, well-defined retrieval metric
in [0, 1] without external dependencies — future revisions replace the
hash rank with a real vector search.
