"""
Structured logging via structlog.

structlog processes log records through a pipeline of processors before
rendering. The stdlib bridge means that standard logging.getLogger() calls
anywhere in this codebase — and in third-party libraries like LangGraph —
flow through the same pipeline and emit the same JSON format.

Call configure_logging() before any other imports in your entry point.
Because structlog intercepts at the record level rather than the handler
level, early log records (e.g. from module-level code in imported packages)
are captured correctly regardless of call order.
"""

import logging
import sys

import structlog


def _add_log_level_uppercase(logger, method_name, event_dict):
    """Add uppercase log level as first field for visual scanning."""
    event_dict["level"] = method_name.upper()
    return event_dict


def configure_logging(level: int = logging.INFO) -> None:
    # Shared processor chain used by both structlog-native and stdlib loggers.
    # Level is first for visual scanning: {"level": "ERROR", ...}
    shared_processors: list[structlog.types.Processor] = [
        _add_log_level_uppercase,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            # Prepare the event dict for the stdlib handler below.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Foreign (stdlib) log records pass through these processors first
        # to normalize them into structlog's event dict format.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
