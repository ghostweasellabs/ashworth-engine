"""
Data Fetcher Agent - Dr. Marcus Thornfield
Senior Market Intelligence Analyst with economist's attention to data quality.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.agents.base import BaseAgent
from src.agents.personality import AgentPersonality
from src.workflows.state_schemas import WorkflowState, AgentStatus, FileProcessingState
from src.utils.file_processors import FileProcessor, FileValidationError, DataExtractionError
from src.models.base import Transaction


class DataFetcherAgent(BaseAgent):
    """
    Dr. Marcus Thornfield - Senior Market Intelligence Analyst
    
    Persona: Wharton PhD and former Federal Reserve economist with analytical rigor
    and rapid synthesis capabilities. Anticipates market shifts and focuses on data
    provenance and completeness with economist's attention to data quality.
    """
    
    def __init__(self):
        personality = AgentPersonality(
            name="Dr. Marcus Thornfield",
            title="Senior Market Intelligence Analyst",
            background="Wharton PhD, former Federal Reserve economist with 15 years of market intelligence experience",
            personality_traits=[
                "Analytical rigor with rapid synthesis capabilities",
                "Anticipates market shifts and economic patterns",
                "Methodical approach to data validation",
                "Focus on data provenance and completeness"
            ],
            communication_style="Methodical and precise, emphasizes data quality and market context",
            expertise_areas=[
                "Financial data extraction and validation",
                "Market context analysis",
                "Data provenance tracking",
                "Economic pattern recognition",
                "Multi-format document processing"
            ],
            system_prompt="""You are Dr. Marcus Thornfield, a Senior Market Intelligence Analyst with a Wharton PhD and former Federal Reserve experience. Your role is to extract and consolidate financial data from various document formats with the analytical rigor of an economist.

Key responsibilities:
- Extract financial data from CSV, Excel, PDF, and image files with robust error handling
- Apply economist's attention to data quality and completeness
- Provide market context annotations and data provenance tracking
- Handle messy real-world data with intelligent recovery strategies
- Maintain data integrity while maximizing extraction success rates

Your approach is methodical and precise, always considering the broader economic context of the data you're processing.""",
            task_prompt_template="""Analyze and extract financial data from the provided files: {file_paths}

Apply your economist's expertise to:
1. Extract all financial transactions with maximum data recovery
2. Assess data quality and identify potential issues
3. Provide market context annotations where relevant
4. Track data provenance for audit purposes
5. Handle inconsistent formats with intelligent normalization

Focus on data completeness while maintaining accuracy standards.""",
            error_handling_style="analytical"
        )
        
        super().__init__(personality)
        self.file_processor = FileProcessor()
        self.max_workers = 4  # For parallel processing
        
    def get_agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return "data_fetcher"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """
        Execute data fetching with parallel processing and comprehensive error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with extracted data
        """
        try:
            # Validate input files
            input_files = state.get("input_files", [])
            if not input_files:
                raise ValueError("No input files provided for data extraction")
            
            self.logger.info(f"Processing {len(input_files)} files with parallel execution")
            
            # Initialize file processing state
            file_processing_state = FileProcessingState(
                status=AgentStatus.IN_PROGRESS,
                validation_errors=[],
                processing_metadata={
                    "total_files": len(input_files),
                    "processing_start": datetime.utcnow().isoformat(),
                    "parallel_workers": self.max_workers
                }
            )
            
            # Process files in parallel
            extracted_data, processing_results = await self._process_files_parallel(input_files)
            
            # Consolidate and validate extracted data
            consolidated_data = self._consolidate_data(extracted_data, processing_results)
            
            # Generate market context annotations
            market_context = self._generate_market_context(consolidated_data)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(processing_results, consolidated_data)
            
            # Update file processing state
            file_processing_state.update({
                "raw_data": consolidated_data,
                "processing_metadata": {
                    **file_processing_state["processing_metadata"],
                    "processing_end": datetime.utcnow().isoformat(),
                    "total_transactions": len(consolidated_data),
                    "quality_score": quality_metrics["overall_quality"],
                    "market_context": market_context,
                    "data_provenance": self._generate_provenance_tracking(processing_results)
                },
                "validation_errors": [result["errors"] for result in processing_results if result["errors"]],
                "status": AgentStatus.COMPLETED
            })
            
            # Update workflow state
            state["file_processing"] = file_processing_state
            
            # Add quality metrics to agent memory
            self.update_memory("quality_metrics", quality_metrics)
            self.update_memory("market_context", market_context)
            
            # Log success with economist's perspective
            self.logger.info(
                f"Data extraction completed. Processed {len(consolidated_data)} transactions "
                f"with {quality_metrics['overall_quality']:.2%} quality score. "
                f"Market context: {market_context['summary']}"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Data fetching failed: {str(e)}")
            # Update file processing state with error
            state["file_processing"] = FileProcessingState(
                status=AgentStatus.FAILED,
                validation_errors=[str(e)],
                processing_metadata={"error_timestamp": datetime.utcnow().isoformat()}
            )
            raise
    
    async def _process_files_parallel(self, file_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process multiple files in parallel with comprehensive error handling.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (extracted_data, processing_results)
        """
        extracted_data = []
        processing_results = []
        
        # Use ThreadPoolExecutor for I/O bound file processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_data, file_result = future.result()
                    extracted_data.extend(file_data)
                    processing_results.append(file_result)
                    
                    self.logger.info(f"Successfully processed {file_path}: {len(file_data)} transactions")
                    
                except Exception as e:
                    error_result = {
                        "file_path": file_path,
                        "success": False,
                        "errors": [str(e)],
                        "warnings": [],
                        "transaction_count": 0,
                        "quality_score": 0.0,
                        "processing_time": 0.0
                    }
                    processing_results.append(error_result)
                    
                    self.logger.warning(f"Failed to process {file_path}: {str(e)}")
        
        return extracted_data, processing_results
    
    def _process_single_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single file with comprehensive error handling and quality assessment.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (extracted_data, processing_result)
        """
        start_time = datetime.utcnow()
        file_path_obj = Path(file_path)
        
        processing_result = {
            "file_path": file_path,
            "file_name": file_path_obj.name,
            "file_size": 0,
            "file_type": file_path_obj.suffix.lower(),
            "success": False,
            "errors": [],
            "warnings": [],
            "transaction_count": 0,
            "quality_score": 0.0,
            "processing_time": 0.0,
            "data_issues": []
        }
        
        try:
            # Validate file exists and get size
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            processing_result["file_size"] = file_path_obj.stat().st_size
            
            # Process file using the file processor
            raw_data = self.file_processor.process_file(file_path)
            
            # Convert to standardized format with quality assessment
            extracted_data = []
            data_issues = []
            
            for i, row in enumerate(raw_data):
                try:
                    # Create transaction with quality assessment
                    transaction_data, quality_issues = self._create_transaction_with_quality_check(
                        row, file_path, i
                    )
                    extracted_data.append(transaction_data)
                    data_issues.extend(quality_issues)
                    
                except Exception as e:
                    warning_msg = f"Row {i+1}: {str(e)}"
                    processing_result["warnings"].append(warning_msg)
                    data_issues.append(f"Row {i+1}: Data extraction failed - {str(e)}")
            
            # Calculate quality metrics
            quality_score = self._calculate_file_quality_score(extracted_data, data_issues, raw_data)
            
            # Update processing result
            processing_result.update({
                "success": True,
                "transaction_count": len(extracted_data),
                "quality_score": quality_score,
                "data_issues": data_issues,
                "processing_time": (datetime.utcnow() - start_time).total_seconds()
            })
            
            return extracted_data, processing_result
            
        except (FileValidationError, DataExtractionError) as e:
            processing_result["errors"].append(f"File processing error: {str(e)}")
            processing_result["processing_time"] = (datetime.utcnow() - start_time).total_seconds()
            return [], processing_result
            
        except Exception as e:
            processing_result["errors"].append(f"Unexpected error: {str(e)}")
            processing_result["processing_time"] = (datetime.utcnow() - start_time).total_seconds()
            return [], processing_result 
   
    def _create_transaction_with_quality_check(
        self, 
        row_data: Dict[str, Any], 
        source_file: str, 
        row_index: int
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Create a transaction from raw data with comprehensive quality checking.
        
        Args:
            row_data: Raw row data from file
            source_file: Source file path
            row_index: Row index for error tracking
            
        Returns:
            Tuple of (transaction_data, quality_issues)
        """
        quality_issues = []
        
        # Generate unique transaction ID
        transaction_id = str(uuid.uuid4())
        
        # Extract and validate required fields with fallbacks
        date_value = self._extract_date_with_fallback(row_data, quality_issues)
        amount_value = self._extract_amount_with_fallback(row_data, quality_issues)
        description = self._extract_description_with_fallback(row_data, quality_issues)
        
        # Extract optional fields
        account_id = self._safe_extract(row_data, ["account", "account_id", "acct"], None)
        counterparty = self._safe_extract(row_data, ["counterparty", "payee", "vendor", "merchant"], None)
        
        # Calculate data quality score for this transaction
        quality_score = self._calculate_transaction_quality_score(
            date_value, amount_value, description, account_id, counterparty, quality_issues
        )
        
        transaction_data = {
            "id": transaction_id,
            "date": date_value.isoformat() if date_value else None,
            "amount": str(amount_value) if amount_value else "0.00",
            "description": description or f"Transaction from {Path(source_file).name} row {row_index + 1}",
            "account_id": account_id,
            "counterparty": counterparty,
            "source_file": source_file,
            "data_quality_score": quality_score,
            "data_issues": quality_issues.copy(),
            "row_index": row_index,
            "extraction_metadata": {
                "extracted_at": datetime.utcnow().isoformat(),
                "raw_data_keys": list(row_data.keys()),
                "data_completeness": self._calculate_completeness(row_data)
            }
        }
        
        return transaction_data, quality_issues
    
    def _extract_date_with_fallback(self, row_data: Dict[str, Any], quality_issues: List[str]) -> Optional[datetime]:
        """Extract date with multiple fallback strategies."""
        date_fields = ["date", "transaction_date", "trans_date", "dt", "timestamp"]
        
        for field in date_fields:
            if field in row_data and row_data[field]:
                try:
                    date_value = row_data[field]
                    
                    # If it's already a datetime object, return it
                    if isinstance(date_value, datetime):
                        return date_value
                    
                    # If it's a string in ISO format, parse it
                    if isinstance(date_value, str):
                        # Try ISO format first
                        if 'T' in date_value:
                            try:
                                return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                            except ValueError:
                                pass
                        
                        # Try multiple date formats
                        date_str = date_value.strip()
                        for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y', '%Y/%m/%d', '%d-%m-%Y']:
                            try:
                                return datetime.strptime(date_str, fmt)
                            except ValueError:
                                continue
                        
                        # If no format worked, record the issue but continue
                        quality_issues.append(f"Unable to parse date format: {date_str}")
                        return None
                    
                except Exception as e:
                    quality_issues.append(f"Date extraction error from field '{field}': {str(e)}")
        
        quality_issues.append("No valid date field found")
        return None
    
    def _extract_amount_with_fallback(self, row_data: Dict[str, Any], quality_issues: List[str]) -> Optional[float]:
        """Extract amount with multiple fallback strategies."""
        amount_fields = ["amount", "value", "total", "sum", "debit", "credit", "balance"]
        
        for field in amount_fields:
            if field in row_data and row_data[field] is not None:
                try:
                    amount_str = str(row_data[field]).strip()
                    
                    # Handle various amount formats
                    if amount_str:
                        # Remove currency symbols, commas, and handle parentheses for negatives
                        cleaned = amount_str.replace('$', '').replace(',', '').replace(' ', '')
                        
                        # Handle parentheses for negative amounts
                        if cleaned.startswith('(') and cleaned.endswith(')'):
                            cleaned = '-' + cleaned[1:-1]
                        
                        # Try to convert to float
                        return float(cleaned)
                        
                except (ValueError, TypeError) as e:
                    quality_issues.append(f"Amount parsing error from field '{field}': {str(e)}")
        
        quality_issues.append("No valid amount field found")
        return None
    
    def _extract_description_with_fallback(self, row_data: Dict[str, Any], quality_issues: List[str]) -> Optional[str]:
        """Extract description with fallback strategies."""
        desc_fields = ["description", "desc", "memo", "reference", "details", "transaction_type"]
        
        for field in desc_fields:
            if field in row_data and row_data[field]:
                desc = str(row_data[field]).strip()
                if desc and desc.lower() not in ['', 'n/a', 'null', 'none']:
                    return desc
        
        quality_issues.append("No meaningful description found")
        return None
    
    def _safe_extract(self, row_data: Dict[str, Any], field_names: List[str], default: Any = None) -> Any:
        """Safely extract a field with multiple possible names."""
        for field in field_names:
            if field in row_data and row_data[field] is not None:
                value = str(row_data[field]).strip()
                if value and value.lower() not in ['', 'n/a', 'null', 'none']:
                    return value
        return default
    
    def _calculate_transaction_quality_score(
        self, 
        date_value: Optional[datetime], 
        amount_value: Optional[float], 
        description: Optional[str],
        account_id: Optional[str],
        counterparty: Optional[str],
        quality_issues: List[str]
    ) -> float:
        """Calculate quality score for a single transaction."""
        score = 0.0
        max_score = 5.0
        
        # Required fields scoring
        if date_value:
            score += 2.0  # Date is critical
        if amount_value is not None:
            score += 2.0  # Amount is critical
        if description:
            score += 0.5  # Description is helpful
        
        # Optional fields scoring
        if account_id:
            score += 0.25
        if counterparty:
            score += 0.25
        
        # Penalty for quality issues
        penalty = min(len(quality_issues) * 0.1, 1.0)
        score = max(0.0, score - penalty)
        
        return score / max_score
    
    def _calculate_completeness(self, row_data: Dict[str, Any]) -> float:
        """Calculate data completeness percentage."""
        if not row_data:
            return 0.0
        
        non_empty_fields = sum(1 for value in row_data.values() 
                              if value is not None and str(value).strip())
        return non_empty_fields / len(row_data)
    
    def _calculate_file_quality_score(
        self, 
        extracted_data: List[Dict[str, Any]], 
        data_issues: List[str], 
        raw_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall quality score for a file."""
        if not extracted_data:
            return 0.0
        
        # Average transaction quality scores
        avg_transaction_quality = sum(
            tx.get("data_quality_score", 0.0) for tx in extracted_data
        ) / len(extracted_data)
        
        # Extraction success rate
        extraction_rate = len(extracted_data) / len(raw_data) if raw_data else 0.0
        
        # Issue penalty
        issue_penalty = min(len(data_issues) * 0.01, 0.3)
        
        overall_score = (avg_transaction_quality * 0.7 + extraction_rate * 0.3) - issue_penalty
        return max(0.0, min(1.0, overall_score))
    
    def _consolidate_data(
        self, 
        extracted_data: List[Dict[str, Any]], 
        processing_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Consolidate extracted data from multiple files with deduplication and validation.
        
        Args:
            extracted_data: All extracted transaction data
            processing_results: Processing results for each file
            
        Returns:
            Consolidated and deduplicated transaction data
        """
        self.logger.info(f"Consolidating {len(extracted_data)} transactions from {len(processing_results)} files")
        
        # Sort by date for better organization
        valid_transactions = [tx for tx in extracted_data if tx.get("date")]
        invalid_transactions = [tx for tx in extracted_data if not tx.get("date")]
        
        # Sort valid transactions by date
        valid_transactions.sort(key=lambda x: x["date"])
        
        # Add economist's market context to each transaction
        for tx in valid_transactions:
            tx["market_context"] = self._add_transaction_market_context(tx)
        
        # Combine valid and invalid transactions
        consolidated = valid_transactions + invalid_transactions
        
        self.logger.info(f"Consolidated to {len(consolidated)} transactions ({len(valid_transactions)} with dates)")
        
        return consolidated
    
    def _add_transaction_market_context(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add market context annotations to a transaction."""
        context = {
            "data_source": Path(transaction["source_file"]).name,
            "extraction_confidence": transaction.get("data_quality_score", 0.0),
            "data_provenance": {
                "extracted_at": transaction.get("extraction_metadata", {}).get("extracted_at"),
                "row_index": transaction.get("row_index"),
                "completeness": transaction.get("extraction_metadata", {}).get("data_completeness", 0.0)
            }
        }
        
        # Add temporal context if date is available
        if transaction.get("date"):
            try:
                tx_date = datetime.fromisoformat(transaction["date"].replace('Z', '+00:00'))
                context["temporal_context"] = {
                    "quarter": f"Q{(tx_date.month - 1) // 3 + 1} {tx_date.year}",
                    "month_name": tx_date.strftime("%B %Y"),
                    "days_from_now": (datetime.utcnow() - tx_date).days
                }
            except Exception:
                pass
        
        return context
    
    def _generate_market_context(self, consolidated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall market context analysis for the dataset."""
        if not consolidated_data:
            return {"summary": "No data available for market context analysis"}
        
        # Analyze temporal distribution
        dated_transactions = [tx for tx in consolidated_data if tx.get("date")]
        
        context = {
            "summary": f"Processed {len(consolidated_data)} transactions with economist's analytical framework",
            "data_coverage": {
                "total_transactions": len(consolidated_data),
                "dated_transactions": len(dated_transactions),
                "coverage_percentage": len(dated_transactions) / len(consolidated_data) * 100 if consolidated_data else 0
            },
            "temporal_analysis": self._analyze_temporal_patterns(dated_transactions),
            "data_quality_assessment": self._assess_overall_data_quality(consolidated_data),
            "economist_insights": self._generate_economist_insights(consolidated_data)
        }
        
        return context
    
    def _analyze_temporal_patterns(self, dated_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in the transaction data."""
        if not dated_transactions:
            return {"pattern": "No dated transactions available"}
        
        try:
            dates = []
            for tx in dated_transactions:
                try:
                    date_obj = datetime.fromisoformat(tx["date"].replace('Z', '+00:00'))
                    dates.append(date_obj)
                except Exception:
                    continue
            
            if not dates:
                return {"pattern": "No valid dates found"}
            
            dates.sort()
            date_range = (dates[-1] - dates[0]).days
            
            return {
                "pattern": "Temporal analysis completed",
                "date_range_days": date_range,
                "earliest_transaction": dates[0].isoformat(),
                "latest_transaction": dates[-1].isoformat(),
                "transaction_frequency": len(dates) / max(date_range, 1) if date_range > 0 else len(dates)
            }
            
        except Exception as e:
            return {"pattern": f"Temporal analysis failed: {str(e)}"}
    
    def _assess_overall_data_quality(self, consolidated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall data quality across all transactions."""
        if not consolidated_data:
            return {"assessment": "No data to assess"}
        
        quality_scores = [tx.get("data_quality_score", 0.0) for tx in consolidated_data]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Count transactions with issues
        transactions_with_issues = sum(1 for tx in consolidated_data if tx.get("data_issues"))
        
        return {
            "assessment": "Data quality analysis completed",
            "average_quality_score": avg_quality,
            "quality_distribution": {
                "high_quality": sum(1 for score in quality_scores if score >= 0.8),
                "medium_quality": sum(1 for score in quality_scores if 0.5 <= score < 0.8),
                "low_quality": sum(1 for score in quality_scores if score < 0.5)
            },
            "transactions_with_issues": transactions_with_issues,
            "issue_rate": transactions_with_issues / len(consolidated_data) * 100
        }
    
    def _generate_economist_insights(self, consolidated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate economist's insights on the data quality and patterns."""
        insights = {
            "data_integrity_assessment": "Applied rigorous data validation protocols",
            "extraction_methodology": "Multi-format processing with intelligent fallback strategies",
            "quality_assurance": "Comprehensive quality scoring and issue tracking implemented"
        }
        
        if consolidated_data:
            # Analyze amount patterns
            amounts = []
            for tx in consolidated_data:
                try:
                    amount = float(tx.get("amount", 0))
                    amounts.append(amount)
                except (ValueError, TypeError):
                    continue
            
            if amounts:
                insights["financial_patterns"] = {
                    "transaction_count": len(amounts),
                    "amount_range": {
                        "min": min(amounts),
                        "max": max(amounts),
                        "avg": sum(amounts) / len(amounts)
                    }
                }
        
        return insights
    
    def _calculate_quality_metrics(
        self, 
        processing_results: List[Dict[str, Any]], 
        consolidated_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for the extraction process."""
        successful_files = sum(1 for result in processing_results if result["success"])
        total_files = len(processing_results)
        
        # Calculate average quality scores
        file_quality_scores = [result.get("quality_score", 0.0) for result in processing_results]
        avg_file_quality = sum(file_quality_scores) / len(file_quality_scores) if file_quality_scores else 0.0
        
        transaction_quality_scores = [tx.get("data_quality_score", 0.0) for tx in consolidated_data]
        avg_transaction_quality = sum(transaction_quality_scores) / len(transaction_quality_scores) if transaction_quality_scores else 0.0
        
        return {
            "overall_quality": (avg_file_quality + avg_transaction_quality) / 2,
            "file_processing": {
                "success_rate": successful_files / total_files if total_files > 0 else 0.0,
                "average_file_quality": avg_file_quality,
                "total_files_processed": total_files,
                "successful_files": successful_files
            },
            "transaction_extraction": {
                "total_transactions": len(consolidated_data),
                "average_transaction_quality": avg_transaction_quality,
                "high_quality_transactions": sum(1 for score in transaction_quality_scores if score >= 0.8),
                "low_quality_transactions": sum(1 for score in transaction_quality_scores if score < 0.5)
            },
            "processing_efficiency": {
                "total_processing_time": sum(result.get("processing_time", 0.0) for result in processing_results),
                "average_file_processing_time": sum(result.get("processing_time", 0.0) for result in processing_results) / total_files if total_files > 0 else 0.0
            }
        }
    
    def _generate_provenance_tracking(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive data provenance tracking information."""
        return {
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "agent_version": "1.0.0",
            "processing_methodology": "Parallel multi-format extraction with quality assessment",
            "file_processing_details": [
                {
                    "file_path": result["file_path"],
                    "file_name": result["file_name"],
                    "file_type": result["file_type"],
                    "file_size": result["file_size"],
                    "success": result["success"],
                    "transaction_count": result["transaction_count"],
                    "quality_score": result["quality_score"],
                    "processing_time": result["processing_time"],
                    "errors": result.get("errors", []),
                    "warnings": result.get("warnings", [])
                }
                for result in processing_results
            ],
            "data_lineage": {
                "source_files": [result["file_path"] for result in processing_results],
                "extraction_agent": "Dr. Marcus Thornfield - Data Fetcher Agent",
                "quality_assurance": "Economist-grade data validation applied"
            }
        }