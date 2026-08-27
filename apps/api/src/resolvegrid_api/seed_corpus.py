"""Bounded seed corpus manifest for Phase 7 Task 6's ingestion pipeline.

The raw Markdown content lives as plain `.md` files under `eval/corpus/`
(sibling to `eval/golden/`, which Task 9's evaluation harness will read
from separately) so the source text is easy to read/diff on its own; this
module is the one place that attaches *ingestion metadata* (source type,
access-scope tags, attribution fields, supersession) to each file, and
loads the file content into a `SeedDocument` ready for
`resolvegrid_api.ingestion.ingest_document`.

Corpus composition (per the task's "4-6 synthetic + 2-3 public" bound):

- 5 synthetic `synthetic_private` Kestrel-internal policies, one per real
  seeded department (`apps/api/src/resolvegrid_api/seed.py`'s
  `_DEPARTMENTS`), tagged with that department's
  `resolvegrid_api.retrieval_authz.normalize_department_tag()` output so a
  real employee's home-department tag actually matches. Two of the five
  are the deliberately-stale pair required by the plan doc: an old VPN
  policy (`status="superseded"`) and its replacement (`status="active"`,
  `supersedes_title` pointing back at the old one).
- 3 public, attributed vendor-style docs (`source_type="public"`,
  `access_scope_tags=[]` -- empty means "visible to everyone," per
  `retrieval_authz.py`'s documented convention). Each carries a `url`,
  `publisher`, `retrieved_at`, and `license` field as the task requires.
  These are short, original explainer paragraphs about real,
  generically-known public technical concepts (VPNs, MFA, SLAs) written
  for this corpus -- NOT scraped from any real external page. `url` uses
  `example.com`, the domain IANA reserves specifically for documentation
  and placeholder use (RFC 2606), so it never impersonates a real
  publisher's real domain. `publisher`/`license` say plainly that this is
  a synthetic placeholder attribution, not a real citation.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Resolved relative to this file (apps/api/src/resolvegrid_api/) rather
# than the process's current working directory, so `ingest_document`
# callers work the same whether invoked from the repo root (`uv run
# --package resolvegrid-api ...`, this repo's normal convention) or from
# inside apps/api. Overridable via SEED_CORPUS_DIR for tests/tooling that
# want to point at a different fixture directory, mirroring
# DATABASE_URL/REDIS_URL's env-override convention elsewhere in this app.
_DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[4] / "eval" / "corpus"
CORPUS_DIR = Path(os.environ.get("SEED_CORPUS_DIR", str(_DEFAULT_CORPUS_DIR)))


@dataclass(frozen=True)
class SeedDocument:
    """One corpus entry's ingestion metadata, plus its loaded raw Markdown.

    `supersedes_title` (not a raw `supersedes_document_id`) names another
    `SeedDocument.title` in this same manifest -- resolving it to a real
    `Document.id` requires that document to already be ingested (it has
    to exist first to be pointed at), so
    `resolvegrid_api.ingestion_worker.run_seed_corpus_ingestion` resolves
    it by processing this list in order and tracking each ingested
    title's resulting `Document.id`. `SEED_CORPUS` below is ordered so
    every `supersedes_title` reference appears earlier in the list than
    the document that uses it.
    """

    filename: str
    title: str
    source_type: str  # "public" | "synthetic_private"
    access_scope_tags: list[str]
    status: str = "active"
    supersedes_title: str | None = None
    url: str | None = None
    publisher: str | None = None
    license: str | None = None
    retrieved_at: datetime | None = None
    effective_date: datetime | None = None
    raw_markdown: str = ""  # populated by load_seed_corpus(), not set literally below


# Department tags below are normalize_department_tag() applied to
# apps/api/src/resolvegrid_api/seed.py's real `_DEPARTMENTS` entries
# ("Platform Engineering", "IT Support", "Security", "People Ops"), so a
# real seeded employee's home-department tag matches these documents.
_SECURITY = "security"
_IT_SUPPORT = "it_support"
_PLATFORM_ENGINEERING = "platform_engineering"
_PEOPLE_OPS = "people_ops"

_PUBLIC_PLACEHOLDER_PUBLISHER = (
    "ResolveGrid Knowledge Base (synthetic placeholder publisher -- these "
    "excerpts are original text written for this corpus, not scraped from "
    "a real external site; see seed_corpus.py's module docstring)"
)
_PUBLIC_PLACEHOLDER_LICENSE = (
    "Author-written summary of a publicly-known general concept; no "
    "external copyrighted source -- treat as freely reproducible within "
    "this synthetic corpus."
)
_RETRIEVED_AT = datetime(2026, 8, 27)

SEED_CORPUS: list[SeedDocument] = [
    # --- Synthetic Kestrel-internal policies -------------------------------
    SeedDocument(
        filename="kestrel-vpn-policy-v1-deprecated.md",
        title="Kestrel VPN Access Policy (v1, deprecated)",
        source_type="synthetic_private",
        access_scope_tags=[_SECURITY],
        status="superseded",
        effective_date=datetime(2023, 1, 1),
    ),
    SeedDocument(
        filename="kestrel-vpn-policy-v2.md",
        title="Kestrel VPN Access Policy (v2)",
        source_type="synthetic_private",
        access_scope_tags=[_SECURITY],
        status="active",
        supersedes_title="Kestrel VPN Access Policy (v1, deprecated)",
        effective_date=datetime(2025, 6, 1),
    ),
    SeedDocument(
        filename="kestrel-laptop-provisioning.md",
        title="Kestrel Laptop Provisioning and Offboarding Policy",
        source_type="synthetic_private",
        access_scope_tags=[_IT_SUPPORT],
    ),
    SeedDocument(
        filename="kestrel-oncall-escalation.md",
        title="Kestrel Platform Engineering On-Call Escalation Policy",
        source_type="synthetic_private",
        access_scope_tags=[_PLATFORM_ENGINEERING],
    ),
    SeedDocument(
        filename="kestrel-time-off-request.md",
        title="Kestrel People Ops Time-Off Request Policy",
        source_type="synthetic_private",
        access_scope_tags=[_PEOPLE_OPS],
    ),
    # --- Attributed public vendor-style docs --------------------------------
    SeedDocument(
        filename="public-what-is-a-vpn.md",
        title="What Is a VPN (Public Reference)",
        source_type="public",
        access_scope_tags=[],
        url="https://example.com/knowledge/what-is-a-vpn",
        publisher=_PUBLIC_PLACEHOLDER_PUBLISHER,
        license=_PUBLIC_PLACEHOLDER_LICENSE,
        retrieved_at=_RETRIEVED_AT,
    ),
    SeedDocument(
        filename="public-what-is-mfa.md",
        title="What Is Multi-Factor Authentication (Public Reference)",
        source_type="public",
        access_scope_tags=[],
        url="https://example.com/knowledge/what-is-mfa",
        publisher=_PUBLIC_PLACEHOLDER_PUBLISHER,
        license=_PUBLIC_PLACEHOLDER_LICENSE,
        retrieved_at=_RETRIEVED_AT,
    ),
    SeedDocument(
        filename="public-what-is-an-sla.md",
        title="What Is a Service-Level Agreement (Public Reference)",
        source_type="public",
        access_scope_tags=[],
        url="https://example.com/knowledge/what-is-an-sla",
        publisher=_PUBLIC_PLACEHOLDER_PUBLISHER,
        license=_PUBLIC_PLACEHOLDER_LICENSE,
        retrieved_at=_RETRIEVED_AT,
    ),
]


def load_seed_corpus() -> list[SeedDocument]:
    """Return `SEED_CORPUS` with each entry's `raw_markdown` populated from
    its file under `CORPUS_DIR`, in manifest order (see `SeedDocument`'s
    docstring for why order matters -- `supersedes_title` references must
    resolve against already-processed entries).
    """
    loaded: list[SeedDocument] = []
    for entry in SEED_CORPUS:
        path = CORPUS_DIR / entry.filename
        raw_markdown = path.read_text(encoding="utf-8")
        loaded.append(
            SeedDocument(
                filename=entry.filename,
                title=entry.title,
                source_type=entry.source_type,
                access_scope_tags=entry.access_scope_tags,
                status=entry.status,
                supersedes_title=entry.supersedes_title,
                url=entry.url,
                publisher=entry.publisher,
                license=entry.license,
                retrieved_at=entry.retrieved_at,
                effective_date=entry.effective_date,
                raw_markdown=raw_markdown,
            )
        )
    return loaded
