from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource


def init_metrics(service_name: str) -> metrics.Meter:
    """Configure the global OTel MeterProvider for this process and return a
    Meter, mirroring `init_tracing`'s idempotent-provider-setup pattern
    exactly (same "if a real provider is already set, reuse it" guard, since
    OTel's API only allows the global provider to be set once per process --
    this also makes repeated calls across test files/lifespan re-entries
    safe, same as `init_tracing`).

    Deliberately uses a `PrometheusMetricReader` (a pull-based reader that
    writes into the process's default `prometheus_client` registry), not an
    OTLP push exporter routed through `otel-collector`. `infra/otel-collector/
    config.yaml`'s Phase 11 Task 6 comment explicitly scoped that collector's
    metrics pipeline to its OWN internal self-telemetry, not application
    metrics, specifically because at that point nothing in apps/api emitted
    OTel metrics yet -- reopening that pipeline just for this one counter
    would be broader-than-necessary. Instead this mirrors the *other* real
    precedent already established in this same phase for exposing
    application-level Prometheus metrics: LiteLLM's own `/metrics` endpoint
    (`infra/litellm/config.yaml`'s `success_callback: ["prometheus"]`),
    scraped directly by Prometheus (`infra/prometheus/prometheus.yml`).
    apps/api's own `/metrics` ASGI endpoint (mounted in `main.py`, backed by
    this same default registry) is scraped the identical way.
    """
    provider = metrics.get_meter_provider()
    if not isinstance(provider, MeterProvider):
        reader = PrometheusMetricReader()
        provider = MeterProvider(
            resource=Resource.create({"service.name": service_name}),
            metric_readers=[reader],
        )
        metrics.set_meter_provider(provider)
    return metrics.get_meter(service_name)
