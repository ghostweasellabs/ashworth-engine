"""LLM performance metrics and monitoring system."""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import statistics

from .base import LLMResponse, ModelTier

logger = logging.getLogger(__name__)


@dataclass
class MetricsEntry:
    """Single metrics entry for LLM performance tracking."""
    timestamp: datetime
    provider: str
    model: str
    tier: str
    prompt_length: int
    response_length: int
    tokens_used: Optional[int]
    response_time: float
    cost: Optional[float]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsEntry':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MetricsLogger:
    """Handles logging and analysis of LLM performance metrics."""
    
    def __init__(self, log_file: str = "logs/llm_metrics.jsonl", enabled: bool = True):
        """Initialize metrics logger."""
        self.log_file = Path(log_file)
        self.enabled = enabled
        self.in_memory_metrics: deque = deque(maxlen=1000)  # Keep last 1000 entries
        
        # Create log directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load recent metrics from file
        self._load_recent_metrics()
    
    def log_request(
        self,
        provider: str,
        model: str,
        tier: ModelTier,
        prompt: str,
        response: Optional[LLMResponse] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Log a single LLM request with metrics."""
        if not self.enabled:
            return
        
        entry = MetricsEntry(
            timestamp=datetime.utcnow(),
            provider=provider,
            model=model,
            tier=tier.value,
            prompt_length=len(prompt),
            response_length=len(response.content) if response else 0,
            tokens_used=response.tokens_used if response else None,
            response_time=response.response_time if response else 0.0,
            cost=response.cost if response else None,
            success=response is not None,
            error_message=str(error) if error else None
        )
        
        # Add to in-memory storage
        self.in_memory_metrics.append(entry)
        
        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write metrics to file: {e}")
    
    def _load_recent_metrics(self) -> None:
        """Load recent metrics from file into memory."""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            entry = MetricsEntry.from_dict(data)
                            self.in_memory_metrics.append(entry)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse metrics line: {e}")
        except Exception as e:
            logger.error(f"Failed to load metrics from file: {e}")
    
    def get_provider_stats(
        self,
        hours: int = 24,
        provider: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics by provider."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter metrics by time and provider
        filtered_metrics = [
            entry for entry in self.in_memory_metrics
            if entry.timestamp >= cutoff_time and (provider is None or entry.provider == provider)
        ]
        
        if not filtered_metrics:
            return {}
        
        # Group by provider
        provider_metrics = defaultdict(list)
        for entry in filtered_metrics:
            provider_metrics[entry.provider].append(entry)
        
        # Calculate statistics for each provider
        stats = {}
        for prov, entries in provider_metrics.items():
            successful_entries = [e for e in entries if e.success]
            
            stats[prov] = {
                "total_requests": len(entries),
                "successful_requests": len(successful_entries),
                "success_rate": len(successful_entries) / len(entries) if entries else 0,
                "avg_response_time": statistics.mean([e.response_time for e in successful_entries]) if successful_entries else 0,
                "median_response_time": statistics.median([e.response_time for e in successful_entries]) if successful_entries else 0,
                "total_cost": sum([e.cost for e in successful_entries if e.cost is not None]),
                "avg_cost_per_request": statistics.mean([e.cost for e in successful_entries if e.cost is not None]) if any(e.cost for e in successful_entries) else 0,
                "total_tokens": sum([e.tokens_used for e in successful_entries if e.tokens_used is not None]),
                "error_count": len(entries) - len(successful_entries),
                "common_errors": self._get_common_errors(entries)
            }
        
        return stats
    
    def get_model_comparison(self, hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different models."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_metrics = [
            entry for entry in self.in_memory_metrics
            if entry.timestamp >= cutoff_time and entry.success
        ]
        
        if not filtered_metrics:
            return {}
        
        # Group by model
        model_metrics = defaultdict(list)
        for entry in filtered_metrics:
            model_key = f"{entry.provider}:{entry.model}"
            model_metrics[model_key].append(entry)
        
        # Calculate comparison metrics
        comparison = {}
        for model, entries in model_metrics.items():
            comparison[model] = {
                "request_count": len(entries),
                "avg_response_time": statistics.mean([e.response_time for e in entries]),
                "avg_cost": statistics.mean([e.cost for e in entries if e.cost is not None]) if any(e.cost for e in entries) else 0,
                "tokens_per_second": self._calculate_tokens_per_second(entries),
                "cost_per_token": self._calculate_cost_per_token(entries),
                "tier_distribution": self._get_tier_distribution(entries)
            }
        
        return comparison
    
    def get_performance_alerts(
        self,
        response_time_threshold: float = 30.0,
        cost_threshold: float = 0.10,
        error_rate_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds."""
        alerts = []
        stats = self.get_provider_stats(hours=1)  # Check last hour
        
        for provider, provider_stats in stats.items():
            # Check response time
            if provider_stats["avg_response_time"] > response_time_threshold:
                alerts.append({
                    "type": "slow_response",
                    "provider": provider,
                    "value": provider_stats["avg_response_time"],
                    "threshold": response_time_threshold,
                    "message": f"{provider} average response time ({provider_stats['avg_response_time']:.2f}s) exceeds threshold ({response_time_threshold}s)"
                })
            
            # Check cost
            if provider_stats["avg_cost_per_request"] > cost_threshold:
                alerts.append({
                    "type": "high_cost",
                    "provider": provider,
                    "value": provider_stats["avg_cost_per_request"],
                    "threshold": cost_threshold,
                    "message": f"{provider} average cost per request (${provider_stats['avg_cost_per_request']:.4f}) exceeds threshold (${cost_threshold})"
                })
            
            # Check error rate
            error_rate = 1 - provider_stats["success_rate"]
            if error_rate > error_rate_threshold:
                alerts.append({
                    "type": "high_error_rate",
                    "provider": provider,
                    "value": error_rate,
                    "threshold": error_rate_threshold,
                    "message": f"{provider} error rate ({error_rate:.2%}) exceeds threshold ({error_rate_threshold:.2%})"
                })
        
        return alerts
    
    def _get_common_errors(self, entries: List[MetricsEntry]) -> List[Dict[str, Any]]:
        """Get most common error messages."""
        error_counts = defaultdict(int)
        for entry in entries:
            if not entry.success and entry.error_message:
                error_counts[entry.error_message] += 1
        
        return [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _calculate_tokens_per_second(self, entries: List[MetricsEntry]) -> float:
        """Calculate average tokens per second."""
        valid_entries = [
            e for e in entries 
            if e.tokens_used is not None and e.response_time > 0
        ]
        
        if not valid_entries:
            return 0.0
        
        rates = [e.tokens_used / e.response_time for e in valid_entries]
        return statistics.mean(rates)
    
    def _calculate_cost_per_token(self, entries: List[MetricsEntry]) -> float:
        """Calculate average cost per token."""
        valid_entries = [
            e for e in entries 
            if e.cost is not None and e.tokens_used is not None and e.tokens_used > 0
        ]
        
        if not valid_entries:
            return 0.0
        
        rates = [e.cost / e.tokens_used for e in valid_entries]
        return statistics.mean(rates)
    
    def _get_tier_distribution(self, entries: List[MetricsEntry]) -> Dict[str, int]:
        """Get distribution of requests by tier."""
        tier_counts = defaultdict(int)
        for entry in entries:
            tier_counts[entry.tier] += 1
        return dict(tier_counts)
    
    def export_metrics(
        self,
        output_file: str,
        hours: int = 24,
        format: str = "json"
    ) -> None:
        """Export metrics to file."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_metrics = [
            entry for entry in self.in_memory_metrics
            if entry.timestamp >= cutoff_time
        ]
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in filtered_metrics], f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if filtered_metrics:
                    writer = csv.DictWriter(f, fieldnames=filtered_metrics[0].to_dict().keys())
                    writer.writeheader()
                    for entry in filtered_metrics:
                        writer.writerow(entry.to_dict())
        else:
            raise ValueError(f"Unsupported format: {format}")


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self, metrics_logger: MetricsLogger, config: Dict[str, Any]):
        """Initialize performance monitor."""
        self.metrics_logger = metrics_logger
        self.config = config
        self.alert_callbacks: List[callable] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        interval = self.config.get("health_check_interval_seconds", 300)
        
        while self._monitoring:
            try:
                # Check for performance alerts
                alerts = self.metrics_logger.get_performance_alerts(
                    response_time_threshold=self.config.get("performance_threshold_seconds", 30.0),
                    cost_threshold=self.config.get("cost_threshold_dollars", 0.10),
                    error_rate_threshold=0.1
                )
                
                # Send alerts to callbacks
                for alert in alerts:
                    for callback in self.alert_callbacks:
                        try:
                            await callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying


# Global metrics logger instance
_metrics_logger: Optional[MetricsLogger] = None


def get_metrics_logger() -> MetricsLogger:
    """Get global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        from ...config.settings import settings
        _metrics_logger = MetricsLogger(
            log_file=settings.metrics_log_file,
            enabled=settings.enable_metrics_logging
        )
    return _metrics_logger


def log_llm_request(
    provider: str,
    model: str,
    tier: ModelTier,
    prompt: str,
    response: Optional[LLMResponse] = None,
    error: Optional[Exception] = None
) -> None:
    """Convenience function to log LLM request."""
    metrics_logger = get_metrics_logger()
    metrics_logger.log_request(provider, model, tier, prompt, response, error)