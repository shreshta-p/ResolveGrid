# Telemetry & Cost

ResolveGrid-owned telemetry/cost schema (`AgentRun`/`Span`/`ModelCall`/`PricingVersion`) is defined in the approved architecture plan (§9). As of Phase 1, only a bare OTel Collector with a `debug` exporter exists (`infra/otel-collector/config.yaml`) — it logs received spans to stdout, nothing is persisted yet. `ModelCall`/`PricingVersion` land in Phase 4; Langfuse/Prometheus/Grafana land alongside later phases per §9.
