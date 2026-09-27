import re
from unittest.mock import patch

import pytest
from sqlalchemy import select

from resolvegrid_api.llm_gateway import CompletionResult, LLMGatewayError
from resolvegrid_api.model_call_logging import log_completion
from resolvegrid_api.models import ModelCall, PricingVersion
from resolvegrid_telemetry import init_tracing


@pytest.fixture(scope="module")
def tracer():
    # Real, working tracer bound to a real (recording) global TracerProvider
    # -- NOT the module's own trace.get_tracer(__name__) used bare, which
    # would be a no-op tracer (trace_id always 0/INVALID) if no test in this
    # pytest session has triggered main.py's lifespan (resolvegrid_telemetry.
    # init_tracing) yet. init_tracing() is idempotent (OTel only allows the
    # global provider to be set once per process), so calling it here
    # guarantees every assertion below exercises a REAL span with a REAL,
    # non-zero trace id regardless of what other test modules do or don't
    # import first.
    #
    # Deliberately a lazy fixture, NOT a bare module-level call: pytest
    # imports every test module during collection, before any test in the
    # whole session runs. A bare `tracer = init_tracing(...)` at module
    # scope would therefore execute during collection -- i.e. before
    # test_health.py's own test_lifespan_starts_and_stops_tracing gets a
    # chance to run and legitimately be the first real caller of
    # init_tracing("resolvegrid-api") -- and since init_tracing is
    # idempotent, whichever call wins that race permanently fixes the
    # process-wide provider's resource.service.name. This bit a real full
    # suite run (test_health.py started failing its
    # `service.name == "resolvegrid-api"` assertion once this test file
    # existed, purely from import-order collection side effects, not
    # execution order). A module-scoped fixture defers the call to actual
    # test-run time, which respects normal execution order instead.
    return init_tracing("test-model-call-logging")


_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _assert_real_trace_id(trace_id: str | None) -> None:
    assert trace_id is not None
    assert _TRACE_ID_RE.fullmatch(trace_id), f"not a 32-char lowercase hex string: {trace_id!r}"
    # The all-zero id is OTel's reserved INVALID_TRACE_ID sentinel (what a
    # no-op/unrecorded span context formats to) -- a genuinely real span
    # from a real recording TracerProvider never produces this.
    assert trace_id != "0" * 32, "trace_id is the reserved invalid/all-zero id, not a real one"


def test_log_completion_success_writes_model_call_with_real_trace_id_and_correct_cost(db_session, tracer):
    pricing = PricingVersion(
        provider="test-provider", model="test-model-success",
        input_cost_per_1k_tokens_usd=2.0, output_cost_per_1k_tokens_usd=4.0,
    )
    db_session.add(pricing)
    db_session.flush()

    fake_result = CompletionResult(
        text="a real-shaped completion", input_tokens=100, output_tokens=200, latency_ms=50,
        provider="test-provider", model="test-model-success",
    )
    with patch("resolvegrid_api.model_call_logging.llm_gateway.complete", return_value=fake_result) as mock_complete:
        result = log_completion(db_session, tracer, purpose="test.success", prompt="summarize this")

    mock_complete.assert_called_once()
    assert result is fake_result

    call = db_session.scalar(
        select(ModelCall).where(ModelCall.purpose == "test.success").order_by(ModelCall.id.desc())
    )
    assert call is not None
    assert call.status == "success"
    assert call.provider == "test-provider"
    assert call.model == "test-model-success"
    assert call.input_tokens == 100
    assert call.output_tokens == 200
    assert call.latency_ms == 50
    assert call.pricing_version_id == pricing.id
    # 100/1000 * 2.0 + 200/1000 * 4.0 == 0.2 + 0.8 == 1.0 -- deliberately
    # distinct input/output costs so this genuinely exercises both terms of
    # the cost formula, not just a single-rate shortcut that would also pass
    # if the formula silently dropped one side.
    assert call.estimated_cost_usd == pytest.approx(1.0)

    _assert_real_trace_id(call.trace_id)


def test_log_completion_error_path_writes_error_row_and_reraises(db_session, tracer):
    with patch(
        "resolvegrid_api.model_call_logging.llm_gateway.complete",
        side_effect=LLMGatewayError("boom"),
    ):
        with pytest.raises(LLMGatewayError, match="boom"):
            log_completion(db_session, tracer, purpose="test.error", prompt="summarize this")

    call = db_session.scalar(
        select(ModelCall).where(ModelCall.purpose == "test.error").order_by(ModelCall.id.desc())
    )
    assert call is not None
    assert call.status == "error"
    assert call.error_message == "boom"
    assert call.pricing_version_id is None
    assert call.input_tokens == 0
    assert call.output_tokens == 0
    assert call.latency_ms == 0
    assert call.estimated_cost_usd == 0.0

    _assert_real_trace_id(call.trace_id)


def test_log_completion_no_pricing_version_falls_back_to_zero_cost(db_session, tracer):
    # No PricingVersion row seeded for this provider/model pair at all --
    # mirrors test_ticket_summarize.py's equivalent case, now exercised
    # directly against the shared helper rather than only through the HTTP
    # endpoint.
    fake_result = CompletionResult(
        text="unpriced completion", input_tokens=10, output_tokens=5, latency_ms=25,
        provider="test-provider-unpriced", model="test-model-unpriced",
    )
    with patch("resolvegrid_api.model_call_logging.llm_gateway.complete", return_value=fake_result):
        log_completion(db_session, tracer, purpose="test.unpriced", prompt="summarize this")

    call = db_session.scalar(
        select(ModelCall).where(ModelCall.purpose == "test.unpriced").order_by(ModelCall.id.desc())
    )
    assert call is not None
    assert call.status == "success"
    assert call.pricing_version_id is None
    assert call.estimated_cost_usd == 0.0
    _assert_real_trace_id(call.trace_id)
