from opentelemetry import trace as otel_trace

from fastapi.testclient import TestClient

from resolvegrid_api.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_starts_and_stops_tracing():
    with TestClient(app) as lifespan_client:
        response = lifespan_client.get("/health")
        assert response.status_code == 200
        provider = otel_trace.get_tracer_provider()
        assert provider.resource.attributes["service.name"] == "resolvegrid-api"
