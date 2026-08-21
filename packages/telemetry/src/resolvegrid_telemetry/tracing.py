import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_tracing(service_name: str) -> trace.Tracer:
    """Configure the global OTel tracer provider for this process and return a Tracer.

    Idempotent: if a real TracerProvider is already set (e.g. an earlier call
    from another module in the same process), returns a Tracer from the
    existing provider instead of silently failing to override it — OTel's API
    only allows the global provider to be set once per process.
    """
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)
