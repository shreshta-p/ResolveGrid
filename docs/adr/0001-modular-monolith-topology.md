# ADR 0001: Modular Monolith Topology

## Status
Accepted

## Context
Plan.md requires "clear bounded modules over premature microservices" and to "add queues/events only where async execution, retries, or isolation require them." Three topology options were weighed: full microservices per bounded module, a single synchronous process, and a modular monolith with targeted async workers.

## Decision
One FastAPI deployable importing `agent-orchestration`, `retrieval`, `authz`, `telemetry`, and `operational-adapters` as Python libraries, plus Redis/Arq workers for the four concrete async needs identified: ingestion, batch evaluation, load-replay, notification fan-out. Durability for HITL approvals comes from LangGraph's `AsyncPostgresSaver` checkpointing, not a queue.

## Consequences
Simpler deployment and debugging at current scale. Revisit if sustained CPU/memory contention appears between API request-handling and agent-graph execution, or agent workers need to scale independently of the HTTP tier.
