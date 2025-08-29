"""
Unit tests for Data Processor Agent (Dexter Blackwood).
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.agents.data_processor import DataProcessorAgent
from src.workflows.state_schemas import WorkflowState, AgentStatus, FileProcessingState
from src.models.base import Transaction


class TestDataProcessorAgent:
    """Test suite for Data Processor Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a Data Processor Agent instance."""
        return DataProcessorAgent()
    
    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw transaction data with various quality issues."""
        return [
            {
                "id": "tx1",
                "date": "2024-01-15",
                "amount": "1234.56",
                "description": "Office supplies purchase",
                "account_id": "ACC001",
                "counterparty": "Office Depot",
                "source_file": "test.xlsx"
            },
            {
                "id": "tx2",
                "date": "01/16/2024",  # Different date format
                "amount": "$2,345.67",  # Currency symbol and comma
                "description": "Software license",
                "source_file": "test.xlsx"
            },
            {
                "id": "tx3",
                "date": "2024-01-17T10:30:00",  # ISO format with time
                "amount": "(500.00)",  # Negative in parentheses
                "description": "Refund processed",
                "counterparty": "Customer ABC",
                "source_file": "test.xlsx"
            },
            {
                # Malformed data requiring recovery
                "date": "invalid_date",
                "amount": "not_a_number",
                "description": "",
                "source_file": "test.xlsx"
            },
            {
                # Missing critical fields
                "description": "Incomplete transaction",
                "source_file": "test.xlsx"
            }
        ]
    
    @pytest.fixture
    def workflow_state_with_data(self, sample_raw_data):
        """Workflow state with sample raw data."""
        return WorkflowState(
            workflow_id="test-workflow",
            workflow_type="financial_analysis",
            file_processing=FileProcessingState(
                status=AgentStatus.COMPLETED,
                raw_data=sample_raw_data,
                validation_errors=[],
                processing_metadata={}
            )
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initialization and configuration."""
        assert agent.get_agent_id() == "data_processor"
        assert agent.personality.name == "Dexter Blackwood, PhD"
        assert "Fraud detection algorithms" in agent.personality.expertise_areas
        assert agent.fraud_thresholds["max_single_amount"] == Decimal("100000.00")
    
    @pytest.mark.asyncio
    async def test_execute_with_valid_data(self, agent, workflow_state_with_data):
        """Test successful execution with valid data."""
        result_state = await agent.execute(workflow_state_with_data)
        
        # Check that analysis state was created
        assert "analysis" in result_state
        analysis = result_state["analysis"]
        assert analysis["status"] == AgentStatus.COMPLETED
        assert len(analysis["transactions"]) > 0
        assert "metrics" in analysis
        assert "anomalies" in analysis
    
    @pytest.mark.asyncio
    async def test_execute_with_no_data(self, agent):
        """Test execution with no raw data."""
        empty_state = WorkflowState(
            workflow_id="test-workflow",
            file_processing=FileProcessingState(
                status=AgentStatus.COMPLETED,
                raw_data=[],
                validation_errors=[],
                processing_metadata={}
            )
        )
        
        with pytest.raises(ValueError, match="No raw data available"):
            await agent.execute(empty_state)
    
    @pytest.mark.asyncio
    async def test_validate_and_normalize_date(self, agent):
        """Test date validation and normalization."""
        # Test various date formats
        test_cases = [
            {"date": "2024-01-15"}, 
            {"transaction_date": "01/15/2024"},
            {"dt": "15/01/2024"},
            {"date": datetime(2024, 1, 15)},  # Already datetime
            {"date": "2024-01-15T10:30:00"},  # ISO with time
        ]
        
        for raw_tx in test_cases:
            result = await agent._validate_and_normalize_date(raw_tx)
            assert result is not None
            assert isinstance(result, datetime)
            assert result.year == 2024
            assert result.month == 1
            assert result.day == 15
    
    @pytest.mark.asyncio
    async def test_validate_and_normalize_date_invalid(self, agent):
        """Test date validation with invalid dates."""
        invalid_cases = [
            {"date": "invalid_date"},
            {"date": ""},
            {"date": None},
            {"date": "2050-01-01"},  # Too far in future
            {},  # No date field
        ]
        
        for raw_tx in invalid_cases:
            result = await agent._validate_and_normalize_date(raw_tx)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_validate_and_normalize_amount(self, agent):
        """Test amount validation and normalization."""
        test_cases = [
            ({"amount": "1234.56"}, Decimal("1234.56")),
            ({"amount": "$1,234.56"}, Decimal("1234.56")),
            ({"amount": "(500.00)"}, Decimal("-500.00")),
            ({"amount": "€1.234,56"}, Decimal("1234.56")),  # European format
            ({"amount": 1234.56}, Decimal("1234.56")),  # Float
            ({"amount": Decimal("1234.56")}, Decimal("1234.56")),  # Already Decimal
        ]
        
        for raw_tx, expected in test_cases:
            result = await agent._validate_and_normalize_amount(raw_tx)
            assert result == expected
    
    @pytest.mark.asyncio
    async def test_validate_and_normalize_amount_invalid(self, agent):
        """Test amount validation with invalid amounts."""
        invalid_cases = [
            {"amount": "not_a_number"},
            {"amount": ""},
            {"amount": None},
            {"amount": "10000000"},  # Too large
            {},  # No amount field
        ]
        
        for raw_tx in invalid_cases:
            result = await agent._validate_and_normalize_amount(raw_tx)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_validate_and_clean_description(self, agent):
        """Test description validation and cleaning."""
        test_cases = [
            ({"description": "Office supplies purchase"}, "Office supplies purchase"),
            ({"desc": "  Software license  "}, "Software license"),
            ({"memo": "Refund@#$%processed"}, "Refundprocessed"),
            ({"description": "A" * 250}, "A" * 197 + "..."),  # Truncation
        ]
        
        for raw_tx, expected in test_cases:
            result = await agent._validate_and_clean_description(raw_tx)
            assert result == expected
    
    @pytest.mark.asyncio
    async def test_validate_and_clean_description_invalid(self, agent):
        """Test description validation with invalid descriptions."""
        invalid_cases = [
            {"description": ""},
            {"description": "AB"},  # Too short
            {"description": None},
            {},  # No description field
        ]
        
        for raw_tx in invalid_cases:
            result = await agent._validate_and_clean_description(raw_tx)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_create_validated_transaction(self, agent):
        """Test creation of validated transaction."""
        raw_tx = {
            "id": "tx1",
            "date": "2024-01-15",
            "amount": "1234.56",
            "description": "Office supplies purchase",
            "account_id": "ACC001",
            "counterparty": "Office Depot",
            "source_file": "test.xlsx"
        }
        
        result = await agent._create_validated_transaction(raw_tx, 0)
        
        assert result is not None
        assert isinstance(result, Transaction)
        assert result.id == "tx1"
        assert result.date.year == 2024
        assert result.amount == Decimal("1234.56")
        assert result.description == "Office supplies purchase"
        assert result.account_id == "ACC001"
        assert result.counterparty == "Office Depot"
        assert result.data_quality_score > 0.8  # High quality
        assert len(result.data_issues) == 0
    
    @pytest.mark.asyncio
    async def test_create_validated_transaction_with_issues(self, agent):
        """Test creation of transaction with data quality issues."""
        raw_tx = {
            "date": "2024-01-15",
            "amount": "0.00",  # Zero amount
            "description": "Valid description",  # Make description valid
            "source_file": "test.xlsx"
        }
        
        result = await agent._create_validated_transaction(raw_tx, 0)
        
        assert result is not None
        assert result.amount == Decimal("0.00")
        assert len(result.data_issues) > 0
        assert "Zero amount transaction" in result.data_issues
        assert result.data_quality_score < 0.8  # Lower quality due to zero amount
    
    @pytest.mark.asyncio
    async def test_attempt_data_recovery(self, agent):
        """Test data recovery for malformed transactions."""
        raw_tx = {
            "invalid_field": "some text with 123.45 in it",
            "another_field": "more text",
            "source_file": "test.xlsx"
        }
        
        result = await agent._attempt_data_recovery(raw_tx, 0, "Test error")
        
        assert result is not None
        assert isinstance(result, Transaction)
        assert result.amount == Decimal("123.45")  # Extracted from text
        assert any("Data recovery applied" in issue for issue in result.data_issues)
        assert result.data_quality_score == 0.1  # Very low quality
    
    def test_detect_anomalies_statistical_outliers(self, agent):
        """Test detection of statistical outliers."""
        # Create transactions with one clear outlier
        transactions = []
        for i in range(10):
            tx = Transaction(
                id=f"tx{i}",
                date=datetime.now(),
                amount=Decimal("100.00"),  # Normal amounts
                description=f"Transaction {i}",
                source_file="test.xlsx",
                data_quality_score=1.0,
                data_issues=[]
            )
            transactions.append(tx)
        
        # Add outlier
        outlier = Transaction(
            id="outlier",
            date=datetime.now(),
            amount=Decimal("10000.00"),  # Outlier amount
            description="Outlier transaction",
            source_file="test.xlsx",
            data_quality_score=1.0,
            data_issues=[]
        )
        transactions.append(outlier)
        
        anomalies = agent._detect_anomalies(transactions)
        
        # Should detect the outlier
        outlier_anomalies = [a for a in anomalies if a["type"] == "statistical_outlier"]
        assert len(outlier_anomalies) > 0
        assert outlier_anomalies[0]["transaction_id"] == "outlier"
    
    def test_detect_duplicate_transactions(self, agent):
        """Test detection of duplicate transactions."""
        # Create duplicate transactions
        tx1 = Transaction(
            id="tx1",
            date=datetime(2024, 1, 15),
            amount=Decimal("100.00"),
            description="Office supplies",
            counterparty="Office Depot",
            source_file="test.xlsx",
            data_quality_score=1.0,
            data_issues=[]
        )
        
        tx2 = Transaction(
            id="tx2",
            date=datetime(2024, 1, 15),
            amount=Decimal("100.00"),
            description="Office supplies",
            counterparty="Office Depot",
            source_file="test.xlsx",
            data_quality_score=1.0,
            data_issues=[]
        )
        
        duplicates = agent._detect_duplicate_transactions([tx1, tx2])
        
        assert len(duplicates) == 1
        assert duplicates[0]["type"] == "potential_duplicate"
        assert duplicates[0]["transaction_id"] == "tx2"
    
    def test_detect_fraud_patterns_round_amounts(self, agent):
        """Test detection of round amount fraud patterns."""
        # Create transactions with mostly round amounts
        transactions = []
        for i in range(10):
            amount = Decimal("100.00") if i < 9 else Decimal("123.45")
            tx = Transaction(
                id=f"tx{i}",
                date=datetime.now(),
                amount=amount,
                description=f"Transaction {i}",
                source_file="test.xlsx",
                data_quality_score=1.0,
                data_issues=[]
            )
            transactions.append(tx)
        
        fraud_indicators = agent._detect_fraud_patterns(transactions)
        
        # Should detect high percentage of round amounts
        round_amount_indicators = [f for f in fraud_indicators if "round amounts" in f]
        assert len(round_amount_indicators) > 0
    
    def test_detect_sequential_patterns(self, agent):
        """Test detection of sequential amount patterns."""
        # Create transactions with arithmetic progression
        transactions = []
        for i in range(5):
            tx = Transaction(
                id=f"tx{i}",
                date=datetime.now(),
                amount=Decimal(f"{100 + i * 10}.00"),  # 100, 110, 120, 130, 140
                description=f"Transaction {i}",
                source_file="test.xlsx",
                data_quality_score=1.0,
                data_issues=[]
            )
            transactions.append(tx)
        
        patterns = agent._detect_sequential_patterns(transactions)
        
        assert len(patterns) > 0
        assert "Sequential amount pattern" in patterns[0]
    
    def test_calculate_quality_metrics(self, agent):
        """Test calculation of quality metrics."""
        # Create sample processed transactions
        processed_transactions = [
            Transaction(
                id="tx1",
                date=datetime.now(),
                amount=Decimal("100.00"),
                description="Good transaction",
                source_file="test.xlsx",
                data_quality_score=0.9,
                data_issues=[]
            ),
            Transaction(
                id="tx2",
                date=datetime.now(),
                amount=Decimal("200.00"),
                description="Another transaction",
                source_file="test.xlsx",
                data_quality_score=0.7,
                data_issues=["Minor issue"]
            )
        ]
        
        raw_data = [{"tx": 1}, {"tx": 2}, {"tx": 3}]  # 3 raw, 2 processed
        
        metrics = agent._calculate_quality_metrics(processed_transactions, raw_data)
        
        assert metrics["recovery_rate"] == 2/3  # 2 out of 3 processed
        assert metrics["overall_quality"] > 0
        assert metrics["processing_stats"]["total_raw_transactions"] == 3
        assert metrics["processing_stats"]["total_processed_transactions"] == 2
        assert metrics["quality_distribution"]["high_quality"] == 1
        assert metrics["quality_distribution"]["medium_quality"] == 1
        assert metrics["quality_distribution"]["low_quality"] == 0
    
    def test_generate_recovery_report(self, agent):
        """Test generation of recovery report."""
        # Create sample data with recovery transactions
        processed_transactions = [
            Transaction(
                id="tx1",
                date=datetime.now(),
                amount=Decimal("100.00"),
                description="Normal transaction",
                source_file="test.xlsx",
                data_quality_score=0.9,
                data_issues=[]
            ),
            Transaction(
                id="tx2",
                date=datetime.now(),
                amount=Decimal("200.00"),
                description="Recovered transaction",
                source_file="test.xlsx",
                data_quality_score=0.1,
                data_issues=["Data recovery applied: Test error"]
            )
        ]
        
        raw_data = [{"tx": 1}, {"tx": 2}]
        
        report = agent._generate_recovery_report(processed_transactions, raw_data)
        
        assert "summary" in report
        assert "recovery_strategies_applied" in report
        assert "data_quality_improvements" in report
        assert "validation_results" in report
        assert report["validation_results"]["passed_validation"] == 1
        assert report["validation_results"]["required_recovery"] == 1
    
    @pytest.mark.asyncio
    async def test_process_transactions_with_validation(self, agent, sample_raw_data):
        """Test processing transactions with comprehensive validation."""
        # Reset quality metrics
        agent.quality_metrics = {
            "total_processed": 0,
            "successful_normalizations": 0,
            "data_recovery_attempts": 0,
            "fraud_flags": 0,
            "anomaly_detections": 0
        }
        
        result = await agent._process_transactions_with_validation(sample_raw_data)
        
        assert len(result) > 0  # Should process some transactions
        assert all(isinstance(tx, Transaction) for tx in result)
        assert agent.quality_metrics["total_processed"] == len(sample_raw_data)
        assert agent.quality_metrics["successful_normalizations"] > 0
    
    def test_identify_data_issues(self, agent):
        """Test identification of data quality issues."""
        raw_tx = {
            "date": "2024-01-15",
            "amount": "0.00",
            "description": "AB",  # Too short
            "field1": None,
            "field2": "",
            "field3": "value"
        }
        
        issues = agent._identify_data_issues(
            raw_tx,
            datetime(2024, 1, 15),
            Decimal("0.00"),
            "AB"
        )
        
        assert "Zero amount transaction" in issues
        assert "Very short description" in issues
        # Note: The empty fields check may not trigger with this specific data
    
    def test_safe_extract_and_clean(self, agent):
        """Test safe field extraction and cleaning."""
        raw_tx = {
            "account_id": "  ACC@#$001  ",
            "account": "ACC002",
            "other_field": "n/a"
        }
        
        result = agent._safe_extract_and_clean(raw_tx, ["account_id", "account"], max_length=10)
        assert result == "ACC001"  # Cleaned and from first matching field
        
        # Test with no valid fields
        result = agent._safe_extract_and_clean(raw_tx, ["nonexistent"], max_length=10)
        assert result is None
    
    def test_calculate_transaction_quality_score(self, agent):
        """Test transaction quality score calculation."""
        # High quality transaction
        score = agent._calculate_transaction_quality_score(
            datetime.now(),
            Decimal("100.00"),
            "Good description",
            "ACC001",
            "Vendor Name",
            {"field1": "value1", "field2": "value2"}
        )
        assert score > 0.8
        
        # Low quality transaction
        score = agent._calculate_transaction_quality_score(
            None,  # No date
            None,  # No amount
            None,  # No description
            None,
            None,
            {}
        )
        assert score < 0.2


if __name__ == "__main__":
    pytest.main([__file__])