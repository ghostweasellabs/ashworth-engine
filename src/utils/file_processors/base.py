"""
Base file processor class with common functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class BaseFileProcessor:
    """Base class for file processors."""
    
    def process(self, file_path: Path, source_name: str) -> List[Dict[str, Any]]:
        """Process a file and return raw transaction data."""
        raise NotImplementedError
    
    def _generate_transaction_id(self, row_data: Dict, row_index: int, source_name: str = "unknown") -> str:
        """Generate a unique transaction ID."""
        data_str = f"{source_name}_{row_index}_{str(row_data)}"
        return f"txn_{hash(data_str) % 1000000:06d}"
    
    def _calculate_data_quality_score(self, issues: List[str], total_fields: int) -> float:
        """Calculate data quality score based on issues found."""
        if total_fields == 0:
            return 0.0
        
        # Weight different types of issues
        issue_weights = {
            'missing_date': 0.3,
            'missing_amount': 0.4,
            'missing_description': 0.2,
            'invalid_date': 0.2,
            'invalid_amount': 0.3,
            'malformed_data': 0.1,
        }
        
        total_penalty = sum(issue_weights.get(issue, 0.1) for issue in issues)
        score = max(0.0, 1.0 - (total_penalty / total_fields))
        return min(1.0, score)