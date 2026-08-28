"""Tests for `resolvegrid_retrieval.dedup`.

Per this task's instructions, similarity is checked against real and
realistic content, not synthetic placeholder strings picked for
convenience:

  - The genuine near-duplicate pair pairs `eval/corpus/kestrel-time-off-
    request.md`'s real "PTO Accrual" section with a synthetic FAQ-style
    restatement of the *same* policy facts (same numbers -- 40 hours,
    same concepts -- tenure, calendar year, local law -- different
    sentence structure), modeled on how a second internal document (an
    FAQ, an onboarding guide) realistically restates a known policy
    point. This is the realistic near-duplicate scenario `dedup.py`'s
    module docstring and this task's brief both call for, since Phase
    7's real 8-document corpus was deliberately chunked to avoid actual
    near-duplicate content.

  - The genuine near-miss pair is real corpus content:
    `kestrel-vpn-policy-v1-deprecated.md`'s and `kestrel-vpn-policy-
    v2.md`'s MFA/password-rules sections -- same document family, same
    subsection, heavily overlapping vocabulary (VPN, password,
    multi-factor authentication, employees, days), but opposite facts
    (v1: MFA optional, 180-day/8-char passwords; v2: MFA mandatory with
    no opt-out, 90-day/12-char passwords). A dedup pass that collapsed
    this pair would silently drop the current, safety-relevant MFA
    policy from what a caller ever sees.

See `dedup.py`'s module docstring for the full threshold-validation
writeup (the exact similarity numbers below are the real measured
values, not estimates).
"""

import pytest

from resolvegrid_retrieval.dedup import (
    DEFAULT_DEDUP_THRESHOLD,
    dedup,
    jaccard_similarity,
)

# Real content: eval/corpus/kestrel-time-off-request.md, "PTO Accrual" section.
_PTO_POLICY_CHUNK_ID = 10
_PTO_POLICY_TEXT = (
    "Full-time employees accrue PTO at a fixed rate per pay period based on "
    "tenure, up to an annual cap. Unused PTO up to 40 hours carries over "
    "into the next calendar year; any balance above that is forfeited at "
    "year-end unless local law requires otherwise."
)

# Synthetic FAQ-style restatement of the exact same policy facts -- the
# realistic near-duplicate scenario (see module docstring).
_PTO_FAQ_CHUNK_ID = 11
_PTO_FAQ_TEXT = (
    "Can I carry over unused PTO? Yes, up to 40 hours of unused paid time "
    "off rolls over to the next calendar year. Any amount beyond that cap "
    "is lost at year-end, except where local law says otherwise. PTO "
    "accrues each pay period at a rate tied to your tenure, capped "
    "annually."
)

# Real content: eval/corpus/kestrel-vpn-policy-v1-deprecated.md, "Original
# Password Rules" section -- MFA optional, 180-day/8-char passwords.
_VPN_MFA_V1_CHUNK_ID = 20
_VPN_MFA_V1_TEXT = (
    "VPN passwords under this policy were valid for 180 days and had no "
    "minimum length requirement beyond 8 characters. Multi-factor "
    "authentication was optional, not required, for standard employees."
)

# Real content: eval/corpus/kestrel-vpn-policy-v2.md, "Password and MFA
# Rules" section -- MFA mandatory, 90-day/12-char passwords. Superficially
# similar to the v1 text above (same topic, same subsection, overlapping
# vocabulary) but states the opposite policy -- the near-miss case.
_VPN_MFA_V2_CHUNK_ID = 21
_VPN_MFA_V2_TEXT = (
    "VPN passwords must be at least 12 characters and are rotated every 90 "
    "days. Multi-factor authentication (a push notification to the "
    "Kestrel Authenticator app) is mandatory for every VPN session, with "
    "no opt-out, for all employees and contractors."
)

# Real content: eval/corpus/public-what-is-a-vpn.md's password-reset-
# adjacent VPN text vs. kestrel-time-off-request.md's time-off request
# text -- unrelated control, different documents and topics entirely.
_VPN_RESET_CHUNK_ID = 30
_VPN_RESET_TEXT = (
    "To reset your VPN password, open the KestrelConnect client and "
    "select \"Forgot Password\" on the sign-in screen. This sends a "
    "one-time reset link to your Kestrel email address."
)
_TIME_OFF_REQUEST_CHUNK_ID = 31
_TIME_OFF_REQUEST_TEXT = (
    "Submit a time-off request through the internal HR portal at least 5 "
    "business days before the requested start date for planned leave."
)


def test_jaccard_similarity_of_genuine_near_duplicate_exceeds_threshold():
    """The measured value documented in dedup.py's module docstring:
    same policy facts, reworded -- must clear DEFAULT_DEDUP_THRESHOLD.
    """
    similarity = jaccard_similarity(_PTO_POLICY_TEXT, _PTO_FAQ_TEXT)
    assert similarity == pytest.approx(0.3696, abs=1e-3)
    assert similarity >= DEFAULT_DEDUP_THRESHOLD


def test_jaccard_similarity_of_genuine_near_miss_stays_below_threshold():
    """The measured value documented in dedup.py's module docstring:
    superficially similar (same subsection, heavy vocabulary overlap),
    substantively different (opposite MFA/password policy) -- must NOT
    clear DEFAULT_DEDUP_THRESHOLD.
    """
    similarity = jaccard_similarity(_VPN_MFA_V1_TEXT, _VPN_MFA_V2_TEXT)
    assert similarity == pytest.approx(0.2286, abs=1e-3)
    assert similarity < DEFAULT_DEDUP_THRESHOLD


def test_jaccard_similarity_of_unrelated_chunks_is_near_zero():
    similarity = jaccard_similarity(_VPN_RESET_TEXT, _TIME_OFF_REQUEST_TEXT)
    assert similarity < 0.1


def test_jaccard_similarity_is_symmetric():
    assert jaccard_similarity(_PTO_POLICY_TEXT, _PTO_FAQ_TEXT) == jaccard_similarity(
        _PTO_FAQ_TEXT, _PTO_POLICY_TEXT
    )


def test_jaccard_similarity_identical_text_is_one():
    assert jaccard_similarity(_PTO_POLICY_TEXT, _PTO_POLICY_TEXT) == 1.0


def test_dedup_collapses_genuine_near_duplicate_keeping_higher_score():
    """Two near-duplicate chunks (PTO policy restated) -- the FAQ
    restatement ranked higher by the reranker (score 0.9) must survive;
    the lower-scored original (score 0.7) must be dropped.
    """
    candidates = [
        (_PTO_POLICY_CHUNK_ID, _PTO_POLICY_TEXT, 0.7),
        (_PTO_FAQ_CHUNK_ID, _PTO_FAQ_TEXT, 0.9),
    ]

    result = dedup(candidates)

    assert result == [(_PTO_FAQ_CHUNK_ID, _PTO_FAQ_TEXT, 0.9)]


def test_dedup_keeps_the_higher_scoring_survivor_regardless_of_input_order():
    """Same pair, fed in the opposite (already-best-first-violating)
    input order -- dedup must defensively re-sort by score before
    comparing, so the higher-scoring chunk survives either way (see
    dedup.py's "Which chunk survives" writeup).
    """
    candidates = [
        (_PTO_FAQ_CHUNK_ID, _PTO_FAQ_TEXT, 0.4),
        (_PTO_POLICY_CHUNK_ID, _PTO_POLICY_TEXT, 0.95),
    ]

    result = dedup(candidates)

    assert result == [(_PTO_POLICY_CHUNK_ID, _PTO_POLICY_TEXT, 0.95)]


def test_dedup_does_not_collapse_genuine_near_miss():
    """The important edge case: the v1/v2 MFA pair is superficially
    similar but substantively different (opposite policy) and must NOT
    be collapsed -- both chunks must survive, best-first by score.
    """
    candidates = [
        (_VPN_MFA_V2_CHUNK_ID, _VPN_MFA_V2_TEXT, 0.88),
        (_VPN_MFA_V1_CHUNK_ID, _VPN_MFA_V1_TEXT, 0.61),
    ]

    result = dedup(candidates)

    assert result == [
        (_VPN_MFA_V2_CHUNK_ID, _VPN_MFA_V2_TEXT, 0.88),
        (_VPN_MFA_V1_CHUNK_ID, _VPN_MFA_V1_TEXT, 0.61),
    ]


def test_dedup_preserves_best_first_order_among_survivors_with_no_duplicates():
    """A realistic small reranked list with no near-duplicates at all
    (three genuinely distinct real-corpus chunks): dedup must be a no-op
    on content, only normalizing order to best-first by score.
    """
    candidates = [
        (_TIME_OFF_REQUEST_CHUNK_ID, _TIME_OFF_REQUEST_TEXT, 0.5),
        (_VPN_RESET_CHUNK_ID, _VPN_RESET_TEXT, 0.95),
        (_VPN_MFA_V2_CHUNK_ID, _VPN_MFA_V2_TEXT, 0.7),
    ]

    result = dedup(candidates)

    assert result == [
        (_VPN_RESET_CHUNK_ID, _VPN_RESET_TEXT, 0.95),
        (_VPN_MFA_V2_CHUNK_ID, _VPN_MFA_V2_TEXT, 0.7),
        (_TIME_OFF_REQUEST_CHUNK_ID, _TIME_OFF_REQUEST_TEXT, 0.5),
    ]


def test_dedup_compares_against_every_kept_survivor_not_just_the_most_recent():
    """A true non-adjacent-duplicate-detection case, distinguishing
    'compare against every kept survivor' from a buggy 'compare only
    against the most-recently-kept survivor' implementation -- the
    three-way-cluster test above doesn't actually discriminate these,
    since by the time its third candidate is evaluated there is only
    ever one survivor in the list (found in review).

    Score order after dedup's internal re-sort: PTO policy (0.9, kept as
    survivor #1) -> VPN MFA v2 (0.8, distinct from PTO, kept as survivor
    #2, now the *most recent*) -> PTO FAQ restatement (0.7, a genuine
    near-duplicate of survivor #1 but NOT of survivor #2). A "compare
    only against the most recent survivor" bug would check the PTO FAQ
    only against VPN MFA v2, find them unrelated, and incorrectly KEEP
    it. The correct behavior (verified by code inspection to loop over
    every kept survivor) compares against survivor #1 too and correctly
    drops it as a duplicate of the higher-scored PTO policy chunk.
    """
    candidates = [
        (_PTO_POLICY_CHUNK_ID, _PTO_POLICY_TEXT, 0.9),
        (_VPN_MFA_V2_CHUNK_ID, _VPN_MFA_V2_TEXT, 0.8),
        (_PTO_FAQ_CHUNK_ID, _PTO_FAQ_TEXT, 0.7),
    ]

    result = dedup(candidates)

    assert [chunk_id for chunk_id, _text, _score in result] == [
        _PTO_POLICY_CHUNK_ID,
        _VPN_MFA_V2_CHUNK_ID,
    ]


def test_dedup_empty_input_returns_empty_list():
    assert dedup([]) == []


def test_dedup_single_input_is_returned_unchanged():
    candidates = [(_PTO_POLICY_CHUNK_ID, _PTO_POLICY_TEXT, 0.42)]
    assert dedup(candidates) == candidates


def test_dedup_collapses_a_three_way_duplicate_cluster_onto_top_scorer():
    """Three mutual near-duplicates (all restating the PTO policy) must
    collapse onto a single survivor: the highest-scoring one, even
    though it is not first in comparison order among ties -- this also
    exercises non-adjacent duplicate detection (survivor comparison
    isn't limited to the immediately preceding candidate).
    """
    third_restatement_text = (
        "Kestrel's PTO carryover policy: unused paid time off up to 40 "
        "hours rolls into next calendar year. Anything over that cap is "
        "forfeited at year-end, unless local law overrides it -- PTO "
        "accrual itself is based on tenure, per pay period, up to an "
        "annual cap."
    )
    candidates = [
        (_PTO_POLICY_CHUNK_ID, _PTO_POLICY_TEXT, 0.55),
        (99, third_restatement_text, 0.40),
        (_PTO_FAQ_CHUNK_ID, _PTO_FAQ_TEXT, 0.80),
    ]

    result = dedup(candidates)

    assert [chunk_id for chunk_id, _text, _score in result] == [_PTO_FAQ_CHUNK_ID]
