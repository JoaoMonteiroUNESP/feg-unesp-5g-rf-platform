"""
Structured logging for the FEG-UNESP RF research platform.

Two output channels
-------------------
* Console: human-readable line per record (works on Windows + ANSI-friendly).
* File   : JSON per record at `data/logs/feg.log` (rotating, 5 MB × 5 files).

Usage
-----
    from app.logging_config import configure_logging, get_logger
    configure_logging()                  # call once at app startup
    log = get_logger(__name__)
    log.info("upload accepted", extra={"sha": sha, "rows": n})

`extra={...}` ends up as top-level keys in the JSON record so log analysis
tools can filter (e.g. by run_id, sha, latency_ms).
"""
from __future__ import annotations
import json
import logging
import logging.config
from datetime import datetime, timezone

from app.config import DATA_DIR


LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "feg.log"


# Built-in attrs we don't want re-emitted as JSON keys.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.fromtimestamp(record.created, tz=timezone.utc)
                                .isoformat(timespec="milliseconds"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        # Promote `extra` kwargs to top-level keys.
        for k, v in record.__dict__.items():
            if k in _RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)[:200]
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure both console + JSON-file handlers. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {"()": JsonFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "level": level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filename": str(LOG_FILE),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
                "level": level,
            },
        },
        "root": {"handlers": ["console", "file"], "level": level},
        # Silence libraries that are too chatty at DEBUG.
        "loggers": {
            "uvicorn.access": {"level": "WARNING"},
            "httpx":          {"level": "WARNING"},
            "httpcore":       {"level": "WARNING"},
        },
    })
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
