"""
Production-grade logging configuration for TechGear Electronics Chatbot
Supports structured logging, multiple handlers, and log rotation
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# =============================================================================
# Configuration
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "./logs/app.log")
LOG_MAX_SIZE_MB = int(os.getenv("LOG_MAX_SIZE_MB", "100"))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "10"))
STRUCTURED_LOGGING = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# =============================================================================
# Custom Formatters
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    JSON structured logging formatter
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "environment": ENVIRONMENT,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for better readability
    """
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors
        
        Args:
            record: Log record
            
        Returns:
            Colored log string
        """
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Build log message
        log_message = (
            f"{color}[{timestamp}] "
            f"[{record.levelname:8s}] "
            f"[{record.name}] "
            f"{record.getMessage()}{reset}"
        )
        
        # Add exception if present
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
        
        return log_message


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(
    name: str = "techgear_chatbot",
    level: str = LOG_LEVEL,
    log_file: str = LOG_FILE_PATH
) -> logging.Logger:
    """
    Setup production logger with multiple handlers
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    if STRUCTURED_LOGGING:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_SIZE_MB * 1024 * 1024,  # Convert MB to bytes
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Context Logger
# =============================================================================

class ContextLogger:
    """
    Logger with context fields for structured logging
    """
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize context logger
        
        Args:
            logger: Base logger
            **context: Context fields to include in all logs
        """
        self.logger = logger
        self.context = context
    
    def _log(self, level: int, message: str, **extra_fields):
        """
        Log with context
        
        Args:
            level: Log level
            message: Log message
            **extra_fields: Additional fields
        """
        all_fields = {**self.context, **extra_fields}
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)", 0,
            message, (), None
        )
        record.extra_fields = all_fields
        
        self.logger.handle(record)
    
    def debug(self, message: str, **extra_fields):
        """Log debug message"""
        self._log(logging.DEBUG, message, **extra_fields)
    
    def info(self, message: str, **extra_fields):
        """Log info message"""
        self._log(logging.INFO, message, **extra_fields)
    
    def warning(self, message: str, **extra_fields):
        """Log warning message"""
        self._log(logging.WARNING, message, **extra_fields)
    
    def error(self, message: str, **extra_fields):
        """Log error message"""
        self._log(logging.ERROR, message, **extra_fields)
    
    def critical(self, message: str, **extra_fields):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **extra_fields)


# =============================================================================
# Request Logger Middleware
# =============================================================================

async def log_request_middleware(request, call_next):
    """
    Middleware to log all API requests
    
    Args:
        request: Incoming request
        call_next: Next middleware/handler
        
    Returns:
        Response
    """
    logger = logging.getLogger("techgear_chatbot.requests")
    
    # Log request
    start_time = datetime.now()
    
    logger.info(
        "Request started",
        extra={
            "extra_fields": {
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent"),
            }
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log response
        logger.info(
            "Request completed",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                }
            }
        )
        
        return response
    
    except Exception as e:
        # Log error
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.error(
            f"Request failed: {str(e)}",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "duration_seconds": duration,
                    "error": str(e),
                }
            },
            exc_info=True
        )
        
        raise


# =============================================================================
# Module-level logger
# =============================================================================

# Create default logger
logger = setup_logger()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test logger
    test_logger = setup_logger("test", "DEBUG")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test context logger
    context_logger = ContextLogger(
        test_logger,
        request_id="123456",
        user_id="user_789"
    )
    
    context_logger.info("Processing request", action="process_query")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        test_logger.exception("Exception occurred")
