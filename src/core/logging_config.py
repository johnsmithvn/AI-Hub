"""
Logging configuration with structured logging and correlation IDs
"""

import sys
import json
from pathlib import Path
from loguru import logger
from .config import settings

def serialize_record(record):
    """Serialize log record to JSON, avoid extra recursion"""
    # Chỉ lấy các key trong extra không phải 'extra' lồng nhau
    safe_extra = {k: v for k, v in record["extra"].items() if k != "extra"}
    return json.dumps({
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "extra": safe_extra
    })

def setup_logging():
    """Configure structured logging with correlation IDs"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors for development
    if settings.DEBUG:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True,
            serialize=False
        )
    else:
        # JSON logging for production
        logger.add(
            sys.stdout,
            level=settings.LOG_LEVEL,
            serialize=True
        )
    
    # File handler for all logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "ai_hub.log",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        level="INFO",
        serialize=True
    )
    
    # Error log file
    logger.add(
        log_dir / "errors.log",
        rotation="1 week",
        retention="12 weeks",
        compression="zip",
        level="ERROR",
        serialize=True
    )
