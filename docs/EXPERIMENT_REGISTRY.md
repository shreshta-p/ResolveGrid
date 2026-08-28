# Experiment Registry

Before/after experiment protocol (fixed dataset + seed, paired comparison, raw results under `eval/results/<experiment-id>/`) is defined in the approved architecture plan (§12). First entries land in Phase 7 (retrieval baseline) onward.

## Phase 7 Task 9 — retrieval baseline (`phase7_retrieval_v1`)

| Field | Value |
|---|---|
| Date | 2026-08-27 |
| Git commit | `1f43bce65865c45c427984748b571f4d72631eac` |
| Golden set | `eval/golden/phase7_retrieval_v1.jsonl` (18 hand-written cases: 14 answerable, 2 adversarial authz-leakage/no-good-match, 2 genuinely-unrelated no-good-match; 3 exercise the deliberately-superseded Kestrel VPN v1/v2 distractor pair) |
| Corpus | `eval/corpus/*.md` via `resolvegrid_api.seed_corpus.SEED_CORPUS` (5 synthetic Kestrel-internal policies incl. the superseded VPN v1/v2 pair, 3 attributed public reference docs) -- 8 documents, 36 chunks total |
| Parser version | `markdown-v1` |
| Chunking version | `heading-aware-v1` |
| Embedding model | `nomic-embed-text` (Ollama `/api/embed`) |
| Embedding version | `v1` |
| k | 5 (see `apps/api/src/resolvegrid_api/eval_retrieval.py` module docstring for why: every answerable golden case has exactly one hand-labeled relevant chunk, and several authz-scoped subsets have as few as 4-9 authorized chunks total, so k=5 stays discriminating without trivially recalling everything) |
| Per-signal search limit | 10 (each of `vector_search`/`lexical_search` retrieves top-10 before RRF fusion, so fusion has enough of each ranked list to combine, then metrics are computed on the fused top-5) |
| Harness | `apps/api/src/resolvegrid_api/eval_retrieval.py` (`run_eval`), run via `uv run --package resolvegrid-api python -m resolvegrid_api.eval_retrieval` against a freshly-ingested corpus (rolled back afterward -- no DB residue) |

**Measured results** (mean over the 14 answerable cases; recall@k/precision@k/MRR/nDCG@k are undefined and excluded for the 4 no-good-match cases, per this task's "a correct 'no good match' isn't penalized as a failure" requirement):

| Metric | Value |
|---|---|
| recall@5 | 1.0000 |
| precision@5 | 0.2000 (mechanical ceiling on this golden set -- every answerable case has exactly 1 relevant chunk, so a full hit contributes exactly 1/5) |
| MRR | 0.9524 |
| nDCG@5 | 0.9643 |
| Adversarial authz leakage (`must_not_appear` violated in any case) | None (0 of 18 cases) |
| VPN v1 distractor outranking the correct v2 answer (any distractor case) | None (0 of 3 distractor cases) |

**Observations, reported honestly (not rounded favorably):**

- Recall@5 and precision@5 are effectively perfect on this corpus: every answerable query's single hand-labeled relevant chunk was retrieved within the top-5, in every case. This is a small-corpus effect, not evidence of a generally excellent retriever -- 36 total chunks (as few as 4-9 per authz-scoped subset) means near-exact lexical/semantic matches are much easier to surface than they would be against a realistically-sized enterprise corpus. This baseline should not be read as "retrieval is solved"; it is a floor to catch regressions, not a ceiling to claim.
- The one case that did NOT rank its correct chunk #1 was "Does Kestrel VPN access grant a flat connection to the entire corporate network?" (relevant chunk: `Kestrel VPN Access Policy (v2)`, ordinal 4, the "Access Scope" section) -- it ranked #3 (RR=0.333, nDCG@5=0.5), the only sub-1.0 case in the whole set. Plausible cause: that section's text has less direct lexical overlap with the query's exact phrasing ("flat connection to the entire corporate network" appears near-verbatim in the source text, but the fused rank still slipped behind two other VPN-policy chunks) -- not investigated further, since Task 9 is scoped to measuring and recording, not tuning retrieval.
- The VPN v1/v2 supersession distractor design worked as intended: across all 3 cases where the superseded v1 policy is a plausible-but-wrong answer (password rotation/MFA, password reset process, client software), v2's current chunk always ranked ahead of v1's superseded chunk. `Document.status`/RRF ranking wasn't given any explicit "prefer active over superseded" boost -- this result comes purely from v2's chunk text being a better lexical/semantic match for how the queries were phrased (asking about "current"/"today" policy), not from any status-aware logic in `retrieval.py`. A query phrased more ambiguously (not signaling "current") might not show the same separation -- worth a future, larger eval once supersession-aware ranking is a real feature, not just an emergent property of this hand-picked query phrasing.
- The two adversarial authz cases (`people_ops` principal asking about VPN password rotation; `it_support` principal asking about VPN client software) correctly returned zero leaked chunks from the security-tagged VPN policies, at any rank in the raw (pre-fusion) search results -- consistent with `test_retrieval.py`'s existing adversarial leakage test, now also exercised through this task's own golden queries rather than only hand-constructed fixtures.
- Regression-guard thresholds in `apps/api/tests/test_eval_retrieval.py` were set a bit below these observed numbers (recall@5 >= 0.85, precision@5 >= 0.15, MRR >= 0.80, nDCG@5 >= 0.80), specifically to tolerate normal `nomic-embed-text` embedding-call nondeterminism across runs without being a tautological assertion. The leakage and distractor checks are hard (zero-tolerance) assertions, not threshold-based, since those are correctness properties this corpus is specifically designed to make checkable exactly.
