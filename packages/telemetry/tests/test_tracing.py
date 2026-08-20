from opentelemetry import trace as otel_trace

from resolvegrid_telemetry import init_tracing


def test_init_tracing_sets_service_name_resource():
    tracer = init_tracing("test-service")
    assert tracer is not None
    provider = otel_trace.get_tracer_provider()
    assert provider.resource.attributes["service.name"] == "test-service"


def test_init_tracing_returns_working_tracer():
    tracer = init_tracing("test-service-2")
    with tracer.start_as_current_span("unit-test-span") as span:
        assert span.is_recording()
