"""
Data Processor Agent - Dexter Blackwood, PhD
Quantitative Data Integrity Analyst with zero tolerance for data quality breaches.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Set
import uuid
import re

from src.agents.base import BaseAgent
from src.agents.personality import AgentPersonality
from src.workflows.state_schemas import WorkflowState, AgentStatus, AnalysisState
from src.utils.file_processors.normalizers import DateNormalizer, AmountNormalizer, DataCleaner
from src.models.base import Transaction


class DataProcessorAgent(BaseAgent):
    """
    Dexter Blackwood, PhD - Quantitative Data Integrity Analyst
    
    Persona: MIT PhD with Goldman Sachs background, methodical and data-driven
    with zero tolerance for data quality breaches. Focuses on validation,
    error detection, and fraud prevention through rigorous data analysis.
    """
    
    def __init__(self):
        personality = AgentPersonality(
            name="Dexter Blackwood, PhD",
            title="Quantitative Data Integrity Analyst",
            background="MIT PhD in Quantitative Finance, former Goldman Sachs risk analyst with 12 years of data integrity experience",
            personality_traits=[
                "Methodical and data-driven approach",
                "Zero tolerance for data quality breaches",
                "Fraud detection expertise",
                "Rigorous validation protocols",
                "Statistical anomaly detection"
            ],
            communication_style="Technical and thorough, emphasizes validation and error detection",
            expertise_areas=[
                "Data cleaning and normalization",
                "Fraud detection algorithms",
                "Anomaly detection and quality assurance",
                "Decimal precision financial calculations",
                "Data recovery and validation strategies",
                "Statistical analysis and outlier detection"
            ],
            system_prompt="""You are Dexter Blackwood, PhD, a Quantitative Data Integrity Analyst with MIT PhD and Goldman Sachs experience. Your role is to clean, normalize, and validate financial data with zero tolerance for quality breaches.

Key responsibilities:
- Apply rigorous data cleaning and normalization protocols
- Implement fraud detection algorithms for suspicious patterns
- Perform anomaly detection with statistical rigor
- Ensure Decimal precision for all financial calculations
- Build comprehensive validation with detailed error reporting
- Handle missing fields and malformed data with intelligent recovery strategies

Your approach is methodical and thorough, always prioritizing data integrity and accuracy over processing speed.""",
            task_prompt_template="""Clean and validate the extracted financial data with rigorous quality controls.

Apply your quantitative expertise to:
1. Clean and normalize all transaction data with Decimal precision
2. Detect anomalies and potential fraud indicators
3. Validate data integrity with comprehensive error reporting
4. Implement data recovery strategies for malformed records
5. Generate quality assurance metrics and confidence scores

Focus on data accuracy and fraud prevention while maximizing data recovery.""",
            error_handling_style="analytical"
        )
        
        super().__init__(personality)
        self.date_normalizer = DateNormalizer()
        self.amount_normalizer = AmountNormalizer()
        self.data_cleaner = DataCleaner()
        
        # Fraud detection thresholds
        self.fraud_thresholds = {
            "max_daily_transactions": 100,
            "max_single_amount": Decimal("100000.00"),
            "suspicious_amount_patterns": [
                Decimal("9999.99"), Decimal("10000.00"), Decimal("4999.99")
            ],
            "min_description_length": 3,
            "max_amount_deviation_std": 3.0
        }
        
        # Quality metrics tracking
        self.quality_metrics = {
            "total_processed": 0,
            "successful_normalizations": 0,
            "data_recovery_attempts": 0,
            "fraud_flags": 0,
            "anomaly_detections": 0
        }
        
    def get_agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return "data_processor"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """
        Execute data processing with comprehensive cleaning, validation, and fraud detection.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with processed data
        """
        try:
            # Validate input data from data fetcher
            file_processing = state.get("file_processing", {})
            raw_data = file_processing.get("raw_data", [])
            
            if not raw_data:
                raise ValueError("No raw data available from data fetcher for processing")
            
            self.logger.info(f"Processing {len(raw_data)} raw transactions with rigorous quality controls")
            
            # Initialize analysis state
            analysis_state = AnalysisState(
                status=AgentStatus.IN_PROGRESS,
                transactions=[],
                categories={},
                metrics={},
                anomalies=[],
                compliance_issues=[],
                tax_implications={}
            )
            
            # Process transactions with comprehensive validation
            processed_transactions = await self._process_transactions_with_validation(raw_data)
            
            # Perform anomaly detection
            anomalies = self._detect_anomalies(processed_transactions)
            
            # Run fraud detection algorithms
            fraud_indicators = self._detect_fraud_patterns(processed_transactions)
            
            # Calculate quality assurance metrics
            quality_metrics = self._calculate_quality_metrics(processed_transactions, raw_data)
            
            # Generate data recovery report
            recovery_report = self._generate_recovery_report(processed_transactions, raw_data)
            
            # Update analysis state
            analysis_state.update({
                "transactions": [tx.model_dump() for tx in processed_transactions],
                "anomalies": anomalies,
                "metrics": {
                    "total_transactions": Decimal(str(len(processed_transactions))),
                    "quality_score": Decimal(str(quality_metrics["overall_quality"])),
                    "recovery_rate": Decimal(str(quality_metrics["recovery_rate"])),
                    "fraud_risk_score": Decimal(str(quality_metrics["fraud_risk_score"]))
                },
                "compliance_issues": fraud_indicators,
                "status": AgentStatus.COMPLETED
            })
            
            # Update workflow state
            state["analysis"] = analysis_state
            
            # Add quality metrics to agent memory
            self.update_memory("quality_metrics", quality_metrics)
            self.update_memory("anomalies", anomalies)
            self.update_memory("fraud_indicators", fraud_indicators)
            self.update_memory("recovery_report", recovery_report)
            
            # Log success with quantitative analyst's perspective
            self.logger.info(
                f"Data processing completed with rigorous validation. "
                f"Processed {len(processed_transactions)} transactions "
                f"(quality: {quality_metrics['overall_quality']:.2%}, "
                f"recovery: {quality_metrics['recovery_rate']:.2%}, "
                f"fraud risk: {quality_metrics['fraud_risk_score']:.3f})"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            # Update analysis state with error
            state["analysis"] = AnalysisState(
                status=AgentStatus.FAILED,
                transactions=[],
                categories={},
                metrics={},
                anomalies=[],
                compliance_issues=[str(e)]
            )
            raise
    
    async def _process_transactions_with_validation(self, raw_data: List[Dict[str, Any]]) -> List[Transaction]:
        """
        Process raw transactions with comprehensive validation and data recovery.
        
        Args:
            raw_data: Raw transaction data from data fetcher
            
        Returns:
            List of validated Transaction objects
        """
        processed_transactions = []
        self.quality_metrics["total_processed"] = len(raw_data)
        
        for i, raw_tx in enumerate(raw_data):
            try:
                # Attempt to create validated transaction
                transaction = await self._create_validated_transaction(raw_tx, i)
                if transaction:
                    processed_transactions.append(transaction)
                    self.quality_metrics["successful_normalizations"] += 1
                else:
                    # Transaction validation failed, attempt data recovery
                    self.logger.warning(f"Transaction validation failed for record {i}, attempting recovery")
                    recovered_tx = await self._attempt_data_recovery(raw_tx, i, "Validation failed")
                    if recovered_tx:
                        processed_transactions.append(recovered_tx)
                        self.quality_metrics["data_recovery_attempts"] += 1
                        self.logger.info(f"Successfully recovered transaction {i}")
                
            except Exception as e:
                self.logger.warning(f"Failed to process transaction {i}: {str(e)}")
                
                # Attempt data recovery
                recovered_tx = await self._attempt_data_recovery(raw_tx, i, str(e))
                if recovered_tx:
                    processed_transactions.append(recovered_tx)
                    self.quality_metrics["data_recovery_attempts"] += 1
                    self.logger.info(f"Successfully recovered transaction {i}")
        
        return processed_transactions
    
    async def _create_validated_transaction(self, raw_tx: Dict[str, Any], index: int) -> Optional[Transaction]:
        """
        Create a validated Transaction object with comprehensive data cleaning.
        
        Args:
            raw_tx: Raw transaction data
            index: Transaction index for tracking
            
        Returns:
            Validated Transaction object or None if validation fails
        """
        try:
            # Extract and validate core fields with rigorous cleaning
            transaction_id = raw_tx.get("id") or str(uuid.uuid4())
            
            # Date validation with multiple format support
            date_value = await self._validate_and_normalize_date(raw_tx)
            if not date_value:
                raise ValueError("Invalid or missing date")
            
            # Amount validation with Decimal precision
            amount_value = await self._validate_and_normalize_amount(raw_tx)
            if amount_value is None:
                raise ValueError("Invalid or missing amount")
            
            # Description cleaning and validation
            description = await self._validate_and_clean_description(raw_tx)
            if not description:
                raise ValueError("Invalid or missing description")
            
            # Optional field extraction with validation
            account_id = self._safe_extract_and_clean(raw_tx, ["account_id", "account"], max_length=50)
            counterparty = self._safe_extract_and_clean(raw_tx, ["counterparty", "payee", "vendor"], max_length=100)
            
            # Calculate data quality score
            quality_score = self._calculate_transaction_quality_score(
                date_value, amount_value, description, account_id, counterparty, raw_tx
            )
            
            # Identify data issues
            data_issues = self._identify_data_issues(raw_tx, date_value, amount_value, description)
            
            # Create Transaction object
            transaction = Transaction(
                id=transaction_id,
                date=date_value,
                amount=amount_value,
                description=description,
                account_id=account_id,
                counterparty=counterparty,
                source_file=raw_tx.get("source_file", "unknown"),
                data_quality_score=quality_score,
                data_issues=data_issues
            )
            
            return transaction
            
        except Exception as e:
            self.logger.debug(f"Transaction validation failed for index {index}: {str(e)}")
            return None
    
    async def _validate_and_normalize_date(self, raw_tx: Dict[str, Any]) -> Optional[datetime]:
        """Validate and normalize date with comprehensive format support."""
        date_fields = ["date", "transaction_date", "trans_date", "dt", "timestamp"]
        
        for field in date_fields:
            if field in raw_tx and raw_tx[field] is not None:
                date_value = raw_tx[field]
                
                # Handle datetime objects
                if isinstance(date_value, datetime):
                    return date_value
                
                # Handle string dates
                if isinstance(date_value, str):
                    normalized_date = self.date_normalizer.parse_date(date_value)
                    if normalized_date:
                        # Validate date is reasonable (not too far in future/past)
                        now = datetime.now()
                        if (now - timedelta(days=365*10)) <= normalized_date <= (now + timedelta(days=365)):
                            return normalized_date
        
        return None
    
    async def _validate_and_normalize_amount(self, raw_tx: Dict[str, Any]) -> Optional[Decimal]:
        """Validate and normalize amount with Decimal precision."""
        amount_fields = ["amount", "value", "total", "sum", "debit", "credit"]
        
        for field in amount_fields:
            if field in raw_tx and raw_tx[field] is not None:
                amount_value = raw_tx[field]
                
                # Handle Decimal objects
                if isinstance(amount_value, Decimal):
                    return amount_value
                
                # Handle numeric types
                if isinstance(amount_value, (int, float)):
                    try:
                        return Decimal(str(amount_value))
                    except InvalidOperation:
                        continue
                
                # Handle string amounts
                if isinstance(amount_value, str):
                    normalized_amount = self.amount_normalizer.parse_amount(amount_value)
                    if normalized_amount is not None:
                        # Validate amount is reasonable
                        if Decimal("-1000000") <= normalized_amount <= Decimal("1000000"):
                            return normalized_amount
        
        return None
    
    async def _validate_and_clean_description(self, raw_tx: Dict[str, Any]) -> Optional[str]:
        """Validate and clean description with comprehensive text processing."""
        desc_fields = ["description", "desc", "memo", "reference", "details", "transaction_type"]
        
        for field in desc_fields:
            if field in raw_tx and raw_tx[field] is not None:
                desc_value = str(raw_tx[field]).strip()
                
                if desc_value and len(desc_value) >= self.fraud_thresholds["min_description_length"]:
                    cleaned_desc = self.data_cleaner.clean_description(desc_value)
                    if cleaned_desc and len(cleaned_desc) >= self.fraud_thresholds["min_description_length"]:
                        return cleaned_desc
        
        return None
    
    def _safe_extract_and_clean(self, raw_tx: Dict[str, Any], field_names: List[str], max_length: int = 100) -> Optional[str]:
        """Safely extract and clean a field with validation."""
        for field in field_names:
            if field in raw_tx and raw_tx[field] is not None:
                value = str(raw_tx[field]).strip()
                if value and value.lower() not in ['', 'n/a', 'null', 'none', 'unknown']:
                    # Clean and truncate
                    cleaned = re.sub(r'[^\w\s\-.,()&]', '', value)
                    if len(cleaned) > max_length:
                        cleaned = cleaned[:max_length-3] + '...'
                    return cleaned
        return None
    
    def _calculate_transaction_quality_score(
        self, 
        date_value: Optional[datetime], 
        amount_value: Optional[Decimal], 
        description: Optional[str],
        account_id: Optional[str],
        counterparty: Optional[str],
        raw_tx: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive quality score for a transaction."""
        score = 0.0
        max_score = 10.0
        
        # Core field scoring (70% of total)
        if date_value:
            score += 3.0  # Date is critical
        if amount_value is not None:
            score += 3.0  # Amount is critical
        if description and len(description) >= 5:
            score += 1.0  # Good description
        
        # Optional field scoring (20% of total)
        if account_id:
            score += 1.0
        if counterparty:
            score += 1.0
        
        # Data completeness scoring (10% of total)
        non_empty_fields = sum(1 for value in raw_tx.values() 
                              if value is not None and str(value).strip())
        completeness_score = min(non_empty_fields / 10.0, 1.0)  # Normalize to max 1.0
        score += completeness_score
        
        return min(score / max_score, 1.0)
    
    def _identify_data_issues(
        self, 
        raw_tx: Dict[str, Any], 
        date_value: Optional[datetime], 
        amount_value: Optional[Decimal], 
        description: Optional[str]
    ) -> List[str]:
        """Identify specific data quality issues."""
        issues = []
        
        # Date issues
        if not date_value:
            issues.append("Missing or invalid date")
        elif date_value > datetime.now():
            issues.append("Future date detected")
        
        # Amount issues
        if amount_value is None:
            issues.append("Missing or invalid amount")
        elif amount_value == Decimal("0"):
            issues.append("Zero amount transaction")
        elif amount_value < Decimal("0"):
            issues.append("Negative amount detected")
        
        # Description issues
        if not description:
            issues.append("Missing or invalid description")
        elif len(description) < 5:
            issues.append("Very short description")
        
        # Data completeness issues
        empty_fields = sum(1 for value in raw_tx.values() 
                          if value is None or str(value).strip() == '')
        if empty_fields > len(raw_tx) * 0.5:
            issues.append("High number of empty fields")
        
        return issues
    
    async def _attempt_data_recovery(self, raw_tx: Dict[str, Any], index: int, error: str) -> Optional[Transaction]:
        """
        Attempt intelligent data recovery for failed transactions.
        
        Args:
            raw_tx: Raw transaction data
            index: Transaction index
            error: Original error message
            
        Returns:
            Recovered Transaction object or None
        """
        try:
            self.logger.info(f"Attempting data recovery for transaction {index}: {error}")
            
            # Generate fallback values
            transaction_id = str(uuid.uuid4())
            
            # Date recovery - use current date as fallback
            date_value = datetime.now()
            
            # Amount recovery - try to extract any numeric value
            amount_value = Decimal("0.00")
            for key, value in raw_tx.items():
                if value is not None:
                    # Try to extract any number from the value
                    str_value = str(value)
                    numbers = re.findall(r'\d+\.?\d*', str_value)
                    if numbers:
                        try:
                            amount_value = Decimal(numbers[0])
                            break
                        except InvalidOperation:
                            continue
            
            # Description recovery - use any available text
            description = f"Recovered transaction from row {index + 1}"
            for key, value in raw_tx.items():
                if value is not None and isinstance(value, str) and len(str(value).strip()) > 3:
                    description = self.data_cleaner.clean_description(str(value))
                    break
            
            # Create recovered transaction with low quality score
            transaction = Transaction(
                id=transaction_id,
                date=date_value,
                amount=amount_value,
                description=description,
                source_file=raw_tx.get("source_file", "unknown"),
                data_quality_score=0.1,  # Very low quality for recovered data
                data_issues=[f"Data recovery applied: {error}", "Fallback values used"]
            )
            
            return transaction
            
        except Exception as recovery_error:
            self.logger.error(f"Data recovery failed for transaction {index}: {str(recovery_error)}")
            return None
    
    def _detect_anomalies(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in transaction data using statistical analysis.
        
        Args:
            transactions: List of processed transactions
            
        Returns:
            List of anomaly reports
        """
        anomalies = []
        
        if not transactions:
            return anomalies
        
        # Extract amounts for statistical analysis
        amounts = [float(tx.amount) for tx in transactions]
        
        if len(amounts) < 3:  # Need minimum data for statistical analysis
            return anomalies
        
        # Calculate statistical measures
        mean_amount = statistics.mean(amounts)
        median_amount = statistics.median(amounts)
        
        try:
            std_amount = statistics.stdev(amounts)
        except statistics.StatisticsError:
            std_amount = 0
        
        # Detect statistical outliers
        for i, tx in enumerate(transactions):
            amount = float(tx.amount)
            
            # Z-score analysis
            if std_amount > 0:
                z_score = abs(amount - mean_amount) / std_amount
                if z_score > self.fraud_thresholds["max_amount_deviation_std"]:
                    anomalies.append({
                        "transaction_id": tx.id,
                        "type": "statistical_outlier",
                        "severity": "high" if z_score > 4 else "medium",
                        "description": f"Amount {amount} is {z_score:.2f} standard deviations from mean",
                        "z_score": z_score,
                        "amount": amount,
                        "mean_amount": mean_amount,
                        "std_amount": std_amount
                    })
                    self.quality_metrics["anomaly_detections"] += 1
        
        # Detect duplicate transactions
        duplicates = self._detect_duplicate_transactions(transactions)
        anomalies.extend(duplicates)
        
        # Detect suspicious patterns
        pattern_anomalies = self._detect_suspicious_patterns(transactions)
        anomalies.extend(pattern_anomalies)
        
        self.logger.info(f"Detected {len(anomalies)} anomalies in transaction data")
        return anomalies
    
    def _detect_duplicate_transactions(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect potential duplicate transactions."""
        duplicates = []
        seen_transactions = {}
        
        for tx in transactions:
            # Create a signature for the transaction
            signature = (
                tx.date.date(),
                tx.amount,
                tx.description[:50].lower().strip(),
                tx.counterparty
            )
            
            if signature in seen_transactions:
                duplicates.append({
                    "transaction_id": tx.id,
                    "type": "potential_duplicate",
                    "severity": "medium",
                    "description": f"Potential duplicate of transaction {seen_transactions[signature]}",
                    "duplicate_of": seen_transactions[signature],
                    "signature": str(signature)
                })
                self.quality_metrics["anomaly_detections"] += 1
            else:
                seen_transactions[signature] = tx.id
        
        return duplicates
    
    def _detect_suspicious_patterns(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect suspicious transaction patterns."""
        suspicious = []
        
        # Group transactions by date
        daily_transactions = {}
        for tx in transactions:
            date_key = tx.date.date()
            if date_key not in daily_transactions:
                daily_transactions[date_key] = []
            daily_transactions[date_key].append(tx)
        
        # Check for excessive daily transactions
        for date, day_txs in daily_transactions.items():
            if len(day_txs) > self.fraud_thresholds["max_daily_transactions"]:
                suspicious.append({
                    "transaction_ids": [tx.id for tx in day_txs],
                    "type": "excessive_daily_volume",
                    "severity": "high",
                    "description": f"Excessive transactions on {date}: {len(day_txs)} transactions",
                    "date": str(date),
                    "transaction_count": len(day_txs)
                })
                self.quality_metrics["anomaly_detections"] += 1
        
        # Check for suspicious amounts
        for tx in transactions:
            if tx.amount in self.fraud_thresholds["suspicious_amount_patterns"]:
                suspicious.append({
                    "transaction_id": tx.id,
                    "type": "suspicious_amount_pattern",
                    "severity": "medium",
                    "description": f"Suspicious amount pattern: {tx.amount}",
                    "amount": float(tx.amount)
                })
                self.quality_metrics["anomaly_detections"] += 1
        
        return suspicious
    
    def _detect_fraud_patterns(self, transactions: List[Transaction]) -> List[str]:
        """
        Detect potential fraud indicators using advanced pattern analysis.
        
        Args:
            transactions: List of processed transactions
            
        Returns:
            List of fraud indicator descriptions
        """
        fraud_indicators = []
        
        if not transactions:
            return fraud_indicators
        
        # Check for round number bias (potential manipulation)
        round_amounts = sum(1 for tx in transactions if float(tx.amount) % 1 == 0)
        round_percentage = round_amounts / len(transactions)
        
        if round_percentage > 0.8:
            fraud_indicators.append(
                f"High percentage of round amounts ({round_percentage:.1%}) - potential manipulation"
            )
            self.quality_metrics["fraud_flags"] += 1
        
        # Check for unusual time patterns
        weekend_transactions = sum(1 for tx in transactions if tx.date.weekday() >= 5)
        weekend_percentage = weekend_transactions / len(transactions)
        
        if weekend_percentage > 0.3:
            fraud_indicators.append(
                f"High percentage of weekend transactions ({weekend_percentage:.1%}) - unusual pattern"
            )
            self.quality_metrics["fraud_flags"] += 1
        
        # Check for sequential transaction patterns
        sequential_patterns = self._detect_sequential_patterns(transactions)
        if sequential_patterns:
            fraud_indicators.extend(sequential_patterns)
        
        # Check for description patterns that might indicate automation
        automation_patterns = self._detect_automation_patterns(transactions)
        if automation_patterns:
            fraud_indicators.extend(automation_patterns)
        
        self.logger.info(f"Detected {len(fraud_indicators)} fraud indicators")
        return fraud_indicators
    
    def _detect_sequential_patterns(self, transactions: List[Transaction]) -> List[str]:
        """Detect sequential patterns that might indicate fraud."""
        patterns = []
        
        # Sort transactions by amount
        sorted_txs = sorted(transactions, key=lambda x: x.amount)
        
        # Look for arithmetic progressions in amounts
        if len(sorted_txs) >= 5:
            for i in range(len(sorted_txs) - 4):
                amounts = [float(tx.amount) for tx in sorted_txs[i:i+5]]
                diffs = [amounts[j+1] - amounts[j] for j in range(4)]
                
                # Check if differences are consistent (arithmetic progression)
                if all(abs(diff - diffs[0]) < 0.01 for diff in diffs) and diffs[0] > 0:
                    patterns.append(
                        f"Sequential amount pattern detected: {amounts} (consistent increment: {diffs[0]:.2f})"
                    )
                    self.quality_metrics["fraud_flags"] += 1
                    break
        
        return patterns
    
    def _detect_automation_patterns(self, transactions: List[Transaction]) -> List[str]:
        """Detect patterns that might indicate automated/scripted transactions."""
        patterns = []
        
        # Check for identical descriptions
        description_counts = {}
        for tx in transactions:
            desc = tx.description.lower().strip()
            description_counts[desc] = description_counts.get(desc, 0) + 1
        
        # Find descriptions that appear too frequently
        for desc, count in description_counts.items():
            if count > len(transactions) * 0.5 and count > 10:
                patterns.append(
                    f"Highly repetitive description pattern: '{desc}' appears {count} times ({count/len(transactions):.1%})"
                )
                self.quality_metrics["fraud_flags"] += 1
        
        # Check for timestamp clustering (transactions happening too close together)
        if len(transactions) > 1:
            sorted_by_time = sorted(transactions, key=lambda x: x.date)
            close_transactions = 0
            
            for i in range(len(sorted_by_time) - 1):
                time_diff = (sorted_by_time[i+1].date - sorted_by_time[i].date).total_seconds()
                if time_diff < 60:  # Less than 1 minute apart
                    close_transactions += 1
            
            if close_transactions > len(transactions) * 0.3:
                patterns.append(
                    f"High frequency of closely timed transactions: {close_transactions} pairs within 1 minute"
                )
                self.quality_metrics["fraud_flags"] += 1
        
        return patterns
    
    def _calculate_quality_metrics(self, processed_transactions: List[Transaction], raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics for the processing operation.
        
        Args:
            processed_transactions: Successfully processed transactions
            raw_data: Original raw data
            
        Returns:
            Dictionary of quality metrics
        """
        total_raw = len(raw_data)
        total_processed = len(processed_transactions)
        
        # Basic processing metrics
        recovery_rate = total_processed / total_raw if total_raw > 0 else 0.0
        
        # Quality score distribution
        if processed_transactions:
            quality_scores = [tx.data_quality_score for tx in processed_transactions]
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            high_quality = sum(1 for score in quality_scores if score >= 0.8)
            medium_quality = sum(1 for score in quality_scores if 0.5 <= score < 0.8)
            low_quality = sum(1 for score in quality_scores if score < 0.5)
        else:
            avg_quality = 0.0
            high_quality = medium_quality = low_quality = 0
        
        # Data issues analysis
        total_issues = sum(len(tx.data_issues) for tx in processed_transactions)
        avg_issues_per_transaction = total_issues / total_processed if total_processed > 0 else 0
        
        # Fraud risk assessment
        fraud_risk_score = (
            self.quality_metrics["fraud_flags"] / max(total_processed, 1) +
            self.quality_metrics["anomaly_detections"] / max(total_processed, 1)
        ) / 2
        
        # Overall quality assessment
        overall_quality = (
            recovery_rate * 0.3 +
            avg_quality * 0.4 +
            (1 - min(fraud_risk_score, 1.0)) * 0.2 +
            (1 - min(avg_issues_per_transaction / 5, 1.0)) * 0.1
        )
        
        return {
            "overall_quality": overall_quality,
            "recovery_rate": recovery_rate,
            "fraud_risk_score": fraud_risk_score,
            "processing_stats": {
                "total_raw_transactions": total_raw,
                "total_processed_transactions": total_processed,
                "successful_normalizations": self.quality_metrics["successful_normalizations"],
                "data_recovery_attempts": self.quality_metrics["data_recovery_attempts"],
                "fraud_flags": self.quality_metrics["fraud_flags"],
                "anomaly_detections": self.quality_metrics["anomaly_detections"]
            },
            "quality_distribution": {
                "high_quality": high_quality,
                "medium_quality": medium_quality,
                "low_quality": low_quality,
                "average_quality_score": avg_quality
            },
            "data_issues": {
                "total_issues": total_issues,
                "average_issues_per_transaction": avg_issues_per_transaction,
                "transactions_with_issues": sum(1 for tx in processed_transactions if tx.data_issues)
            }
        }
    
    def _generate_recovery_report(self, processed_transactions: List[Transaction], raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive data recovery and processing report.
        
        Args:
            processed_transactions: Successfully processed transactions
            raw_data: Original raw data
            
        Returns:
            Data recovery report
        """
        recovery_report = {
            "summary": f"Processed {len(processed_transactions)} of {len(raw_data)} raw transactions",
            "recovery_strategies_applied": [],
            "data_quality_improvements": [],
            "validation_results": {
                "passed_validation": 0,
                "required_recovery": 0,
                "failed_processing": len(raw_data) - len(processed_transactions)
            },
            "recommendations": []
        }
        
        # Analyze recovery strategies used
        recovery_transactions = [tx for tx in processed_transactions if "Data recovery applied" in str(tx.data_issues)]
        if recovery_transactions:
            recovery_report["recovery_strategies_applied"].append(
                f"Applied data recovery to {len(recovery_transactions)} transactions"
            )
            recovery_report["validation_results"]["required_recovery"] = len(recovery_transactions)
        
        recovery_report["validation_results"]["passed_validation"] = len(processed_transactions) - len(recovery_transactions)
        
        # Analyze data quality improvements
        if processed_transactions:
            # Date normalization improvements
            date_improvements = sum(1 for tx in processed_transactions if tx.date)
            recovery_report["data_quality_improvements"].append(
                f"Successfully normalized {date_improvements} dates with multiple format support"
            )
            
            # Amount precision improvements
            decimal_amounts = sum(1 for tx in processed_transactions if isinstance(tx.amount, Decimal))
            recovery_report["data_quality_improvements"].append(
                f"Applied Decimal precision to {decimal_amounts} amounts for financial accuracy"
            )
            
            # Description cleaning improvements
            clean_descriptions = sum(1 for tx in processed_transactions if tx.description and len(tx.description) > 5)
            recovery_report["data_quality_improvements"].append(
                f"Cleaned and validated {clean_descriptions} transaction descriptions"
            )
        
        # Generate recommendations
        if len(processed_transactions) < len(raw_data) * 0.9:
            recovery_report["recommendations"].append(
                "Consider reviewing data source quality - high failure rate detected"
            )
        
        if self.quality_metrics["fraud_flags"] > 0:
            recovery_report["recommendations"].append(
                f"Review {self.quality_metrics['fraud_flags']} fraud indicators detected during processing"
            )
        
        if self.quality_metrics["anomaly_detections"] > len(processed_transactions) * 0.1:
            recovery_report["recommendations"].append(
                "High anomaly detection rate - consider additional data validation"
            )
        
        return recovery_report