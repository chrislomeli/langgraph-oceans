"""
world-simulator.agents.commons.node_metrics

Observability abstraction for node execution metrics.

Provides a pluggable metrics interface (NodeMetrics) with a console-logging
implementation (ConsoleMetrics). Production deployments can swap in an
OTel-backed implementation without touching the node_executor code.

The interface mirrors OpenTelemetry's span/histogram conventions:
  - record_duration: latency histogram with status label
  - record_error: error counter with exception type
"""

import json
from abc import ABC, abstractmethod

import structlog

logger = structlog.get_logger(__name__)


class NodeMetrics(ABC):
    """Observability contract for node execution metrics.
    Mirrors the OTEL span/histogram interface without the dependency.
    """

    @abstractmethod
    def record_duration(
        self,
        node: str,
        session_id: str,
        elapsed_ms: float,
        status: str,
    ) -> None: ...

    @abstractmethod
    def record_error(
        self,
        node: str,
        session_id: str,
        elapsed_ms: float,
        error: str,
    ) -> None: ...


class ConsoleMetrics(NodeMetrics):
    """
    Console-logging implementation of NodeMetrics.

    Logs node completion events via structlog. Useful for local development
    and debugging. For production, replace with a metrics exporter that
    writes to Prometheus, Datadog, or your observability stack.
    """

    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    def record_duration(self, node, session_id, elapsed_ms, status) -> None:
        color = self.GREEN if status == "ok" else self.RED
        message = json.dumps(
            dict(
                node=node,
                session_id=session_id,
                elapsed_ms=elapsed_ms,
                status=str(status),
            )
        )
        print(f"{color}● {message}{self.RESET}")

    def record_error(self, node, session_id, elapsed_ms, error) -> None:
        message = json.dumps(
            dict(
                node=node,
                session_id=session_id,
                elapsed_ms=elapsed_ms,
                error=str(error),
            )
        )
        print(f"{self.RED}● {message}{self.RESET}")


# Default — swap this out for OTELMetrics in the future - or not
metrics: NodeMetrics = ConsoleMetrics()
