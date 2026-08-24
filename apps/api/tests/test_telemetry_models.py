from resolvegrid_api.models import ModelCall, PricingVersion


def test_model_call_round_trips_with_pricing_version(db_session):
    pricing = PricingVersion(
        provider="ollama", model="qwen3:14b",
        input_cost_per_1k_tokens_usd=0.0, output_cost_per_1k_tokens_usd=0.0,
    )
    db_session.add(pricing)
    db_session.flush()

    call = ModelCall(
        purpose="ticket.summarize", provider="ollama", model="qwen3:14b",
        pricing_version_id=pricing.id, input_tokens=128, output_tokens=52,
        latency_ms=340, estimated_cost_usd=0.0, status="success",
    )
    db_session.add(call)
    db_session.flush()

    fetched_call = db_session.get(ModelCall, call.id)
    fetched_pricing = db_session.get(PricingVersion, pricing.id)

    assert fetched_call is not None
    assert fetched_call.purpose == "ticket.summarize"
    assert fetched_call.provider == "ollama"
    assert fetched_call.model == "qwen3:14b"
    assert fetched_call.pricing_version_id == fetched_pricing.id
    assert fetched_call.input_tokens == 128
    assert fetched_call.output_tokens == 52
    assert fetched_call.latency_ms == 340
    assert fetched_call.estimated_cost_usd == 0.0
    assert fetched_call.status == "success"
    assert fetched_call.error_message is None

    assert fetched_pricing is not None
    assert fetched_pricing.provider == "ollama"
    assert fetched_pricing.model == "qwen3:14b"
    assert fetched_pricing.input_cost_per_1k_tokens_usd == 0.0
    assert fetched_pricing.output_cost_per_1k_tokens_usd == 0.0


def test_model_call_keeps_pointing_at_original_pricing_version_after_repricing(db_session):
    """Two PricingVersion rows can coexist for the same provider/model.

    A ModelCall keeps pointing at whichever PricingVersion was current when
    it was written, so historical cost attribution never silently shifts
    when a newer price is added later (approved architecture plan §9).
    """
    original_pricing = PricingVersion(
        provider="openai", model="gpt-4o-mini",
        input_cost_per_1k_tokens_usd=0.00015, output_cost_per_1k_tokens_usd=0.0006,
    )
    db_session.add(original_pricing)
    db_session.flush()

    call = ModelCall(
        purpose="ticket.summarize", provider="openai", model="gpt-4o-mini",
        pricing_version_id=original_pricing.id, input_tokens=200, output_tokens=80,
        latency_ms=900, estimated_cost_usd=0.000078, status="success",
    )
    db_session.add(call)
    db_session.flush()

    # A newer PricingVersion row for the SAME provider/model is added later
    # (e.g. the provider changed its published rates).
    newer_pricing = PricingVersion(
        provider="openai", model="gpt-4o-mini",
        input_cost_per_1k_tokens_usd=0.0003, output_cost_per_1k_tokens_usd=0.0012,
    )
    db_session.add(newer_pricing)
    db_session.flush()

    assert newer_pricing.id != original_pricing.id

    fetched_call = db_session.get(ModelCall, call.id)
    assert fetched_call.pricing_version_id == original_pricing.id
    assert fetched_call.pricing_version_id != newer_pricing.id
