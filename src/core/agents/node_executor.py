"""
world-simulator.agents.commons.node_executor

Decorator that wraps every LangGraph node function with cross-cutting concerns:
  - Metrics: records duration and outcome (success/error) per node
  - Error handling: catches exceptions, converts to NodeError, sets state.status = ERROR
  - Tracing: captures session_id for distributed request tracking

The decorator handles both sync and async node functions transparently.
Nodes wrapped with this don't need their own try/except or timer logic.

Usage:
    @node_executor("ingest")
    def ingest_node(state: ClusterAgentState) -> dict:
        # your logic here
        return {"status": StatusValue.PROCESSING}
"""

import asyncio
import traceback
from collections.abc import Callable, Coroutine
from functools import wraps
from time import perf_counter
from typing import Any

from core.agents.node_metrics import metrics
from core.agents.node_types import NodeError, TracedState
from core.agents.state_types import StatusValue


def node_executor(node_name: str | None = None):
    """
    Decorator factory that wraps node functions with metrics + error handling.

    Parameters
    ----------
    node_name : str | None
        Logical name for this node in metrics/logs. If None, uses function.__name__.

    Returns
    -------
    Callable
        A decorator that wraps the node function, handling both sync and async variants.

    Behavior
    --------
    - Records start time on entry
    - On success: records duration with status "ok"
    - On exception: records duration with status "error", captures traceback,
      returns {"status": StatusValue.ERROR, "error": NodeError(...)}
    - Preserves function signature via @wraps for introspection
    """

    def decorator(
        func: Callable[..., dict | Coroutine[Any, Any, dict]],
    ) -> Callable[..., dict | Coroutine[Any, Any, dict]]:
        name = node_name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(state: TracedState, **kwargs) -> dict:
                # LangGraph can pass (state, store=..., config=...). We only need state.
                start = perf_counter()
                session_id = (
                    state.get("session_id") if isinstance(state, dict)
                    else getattr(state, "session_id", None)
                ) or f"<{type(state).__name__}>"

                try:
                    result = await func(state, **kwargs)
                    metrics.record_duration(
                        name, session_id, round((perf_counter() - start) * 1000), "ok"
                    )
                    return result

                except Exception as e:
                    metrics.record_error(
                        name, session_id, round((perf_counter() - start) * 1000), str(e)
                    )
                    error = NodeError(
                        message=str(e),
                        traceback=traceback.format_exc(),
                        node=name,
                        session_id=session_id,
                    )
                    return {
                        "status": StatusValue.ERROR,
                        "error": error,
                    }

            return async_wrapper

        @wraps(func)
        def wrapper(state, **kwargs) -> dict:
            # LangGraph can pass (state, store=..., config=...). We only need state.
            start = perf_counter()
            session_id = (
                state.get("session_id") if isinstance(state, dict)
                else getattr(state, "session_id", None)
            ) or f"<{type(state).__name__}>"

            try:
                result = func(state, **kwargs)
                metrics.record_duration(
                    name, session_id, round((perf_counter() - start) * 1000), "ok"
                )
                return result

            except Exception as e:
                metrics.record_error(
                    name, session_id, round((perf_counter() - start) * 1000), str(e)
                )
                error = NodeError(
                    message=str(e),
                    traceback=traceback.format_exc(),
                    node=name,
                    session_id=session_id,
                )
                return {
                    "status": StatusValue.ERROR,
                    "error": error,
                }

        return wrapper

    return decorator
