"""Tests for `resolvegrid_retrieval.status_adjustment`.

Uses the real, exact scores from `docs/EXPERIMENT_REGISTRY.md`'s Phase 8
Task 6 measurement (`0.989385` for the deprecated VPN v1 client-software
chunk, `0.980007` for the current VPN v2 chunk) -- the actual thin,
non-robust margin that flipped in production, not a synthetic gap picked
to make the fix look good.
"""

from resolvegrid_retrieval.status_adjustment import (
    DEFAULT_SUPERSEDED_PENALTY,
    apply_status_adjustment,
)

_V1_CHUNK_ID = 201  # "Kestrel VPN Access Policy (v1, deprecated)"
_V2_CHUNK_ID = 202  # "Kestrel VPN Access Policy (v2)"


def test_fixes_the_real_measured_vpn_v1_v2_flip():
    # Real measured scores from the Task 6 regression: v1 beats v2 by a
    # thin 0.009385 margin before any status adjustment.
    ranked = [
        (_V1_CHUNK_ID, "v1 client software text", 0.989385),
        (_V2_CHUNK_ID, "v2 client software text", 0.980007),
    ]
    result = apply_status_adjustment(ranked, superseded_chunk_ids={_V1_CHUNK_ID})
    assert [chunk_id for chunk_id, _text, _score in result] == [_V2_CHUNK_ID, _V1_CHUNK_ID]


def test_guarantees_correct_order_regardless_of_margin_width():
    # Even an extreme, maximally-adversarial margin (superseded scores
    # 1.0, non-superseded scores near 0.0) must still resolve correctly --
    # proving this is a strict guarantee, not a threshold tuned to the one
    # measured case above.
    ranked = [
        (1, "superseded, very high raw score", 1.0),
        (2, "active, very low raw score", 0.01),
    ]
    result = apply_status_adjustment(ranked, superseded_chunk_ids={1})
    assert [chunk_id for chunk_id, _text, _score in result] == [2, 1]


def test_no_superseded_candidates_is_a_no_op_reorder():
    ranked = [(1, "a", 0.5), (2, "b", 0.9), (3, "c", 0.1)]
    result = apply_status_adjustment(ranked, superseded_chunk_ids=frozenset())
    assert [chunk_id for chunk_id, _text, _score in result] == [2, 1, 3]


def test_deprioritizes_not_filters_superseded_chunk_still_present():
    ranked = [(1, "superseded", 0.9), (2, "active", 0.5)]
    result = apply_status_adjustment(ranked, superseded_chunk_ids={1})
    ids = [chunk_id for chunk_id, _text, _score in result]
    assert set(ids) == {1, 2}
    assert ids == [2, 1]


def test_two_superseded_chunks_keep_relative_order_among_themselves():
    ranked = [
        (1, "superseded high", 0.9),
        (2, "superseded low", 0.2),
        (3, "active", 0.5),
    ]
    result = apply_status_adjustment(ranked, superseded_chunk_ids={1, 2})
    ids = [chunk_id for chunk_id, _text, _score in result]
    # Active chunk first, then the two superseded ones in their own
    # original best-first order (1 was ranked above 2 before the penalty).
    assert ids == [3, 1, 2]


def test_default_penalty_exceeds_the_full_sigmoid_score_range():
    # rerank() scores are documented as sigmoid-activated, i.e. in [0, 1] --
    # DEFAULT_SUPERSEDED_PENALTY must exceed that whole range so the
    # guarantee in the module docstring actually holds for any real score.
    assert DEFAULT_SUPERSEDED_PENALTY >= 1.0


def test_empty_input_returns_empty():
    assert apply_status_adjustment([], superseded_chunk_ids={1}) == []
