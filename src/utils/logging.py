import json
import logging
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    """Structured JSON logger for agent activities"""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_agent_activity(self, agent_name: str, activity: str, 
                          trace_id: str, **kwargs):
        """Log agent activity with structured format"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            "agent": agent_name,
            "activity": activity,
            **kwargs
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, agent_name: str, error: str, trace_id: str, **kwargs):
        """Log error with context"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            "agent": agent_name,
            "level": "ERROR",
            "error": error,
            **kwargs
        }
        self.logger.error(json.dumps(log_entry))

def get_logger(name: str = __name__) -> logging.Logger:
    """Get a standard logger instance"""
    return logging.getLogger(name)