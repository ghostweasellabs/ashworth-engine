"""
Integration tests for Data Processor Agent with real messy Excel data.
"""

import pytest
import asyncio
from pathlib import Path
from decimal import Decimal

from src.agents.data_processor import DataProcessorAgent
from src.workflows.state_schemas import WorkflowState, AgentStatus, FileProcessingState
from src.utils.file_processors import FileProcessor


class TestDataProcessorIntegration:
    """Integration test suite for Data Processor Agent with real data."""
    
    @pytest.fixture
    def agent(self):
        """Create a Data Processor Agent instance."""
        return DataProcessorAgent()
    
    @pytest.fixture
    def file_processor(self):
        """Create a file processor instance."""
        return FileProcessor()
    
    def test_real_excel_files_exist(self):
        """Test that real Excel files exist for testing."""
        data_dir = Path("data/private")
        excel_files = list(data_dir.glob("*.xlsx"))
        assert len(excel_files) > 0, "No Excel files found in data/private directory"
        
        for file_path in excel_files:
            assert file_path.exists(), f"File {file_path} does not exist"
            assert file_path.stat().st_size > 0, f"File {file_path} is empty"
    
    @pytest.mark.asyncio
    async def test_process_real_excel_data(self, agent, file_processor):
        """Test processing real messy Excel data from data/private folder."""
        data_dir = Path("data/private")
        excel_files = list(data_dir.glob("*.xlsx"))
        
        if not excel_files:
            pytest.skip("No Excel files found in data/private directory")
        
        # Process the first Excel file
        test_file = excel_files[0]
        print(f"\nTesting with file: {test_file}")
        
        try:
            # Extract raw data using file processor
            raw_data = file_processor.process_file(test_file)
            print(f"Extracted {len(raw_data)} raw records")
            
            # Create workflow state with raw data
            workflow_state = WorkflowState(
                workflow_id="integration-test",
                workflow_type="financial_analysis",
                file_processing=FileProcessingState(
                    status=AgentStatus.COMPLETED,
                    raw_data=raw_data,
                    validation_errors=[],
                    processing_metadata={"source_file": str(test_file)}
                )
            )
            
            # Process data with the agent
            result_state = await agent.execute(workflow_state)
            
            # Verify processing results
            assert "analysis" in result_state
            analysis = result_state["analysis"]
            assert analysis["status"] == AgentStatus.COMPLETED
            
            processed_transactions = analysis["transactions"]
            print(f"Successfully processed {len(processed_transactions)} transactions")
            
            # Verify we processed some data
            assert len(processed_transactions) > 0, "No transactions were processed"
            
            # Verify quality metrics
            metrics = analysis["metrics"]
            assert "quality_score" in metrics
            assert "recovery_rate" in metrics
            assert "fraud_risk_score" in metrics
            
            quality_score = float(metrics["quality_score"])
            recovery_rate = float(metrics["recovery_rate"])
            fraud_risk = float(metrics["fraud_risk_score"])
            
            print(f"Quality Score: {quality_score:.2%}")
            print(f"Recovery Rate: {recovery_rate:.2%}")
            print(f"Fraud Risk Score: {fraud_risk:.3f}")
            
            # Basic quality assertions
            assert 0 <= quality_score <= 1, "Quality score should be between 0 and 1"
            assert 0 <= recovery_rate <= 1, "Recovery rate should be between 0 and 1"
            assert fraud_risk >= 0, "Fraud risk score should be non-negative"
            
            # Verify anomaly detection ran
            anomalies = analysis["anomalies"]
            print(f"Detected {len(anomalies)} anomalies")
            
            # Verify compliance issues (fraud indicators) were checked
            compliance_issues = analysis["compliance_issues"]
            print(f"Found {len(compliance_issues)} compliance issues")
            
            # Test individual transaction quality
            high_quality_count = sum(1 for tx in processed_transactions 
                                   if tx.get("data_quality_score", 0) >= 0.8)
            medium_quality_count = sum(1 for tx in processed_transactions 
                                     if 0.5 <= tx.get("data_quality_score", 0) < 0.8)
            low_quality_count = sum(1 for tx in processed_transactions 
                                  if tx.get("data_quality_score", 0) < 0.5)
            
            print(f"Quality Distribution:")
            print(f"  High Quality (≥80%): {high_quality_count}")
            print(f"  Medium Quality (50-79%): {medium_quality_count}")
            print(f"  Low Quality (<50%): {low_quality_count}")
            
            # Verify data recovery capabilities
            recovery_report = agent.get_memory("recovery_report", {})
            if recovery_report:
                print(f"Recovery Report: {recovery_report.get('summary', 'No summary')}")
            
        except Exception as e:
            pytest.fail(f"Failed to process real Excel data: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_process_all_excel_files(self, agent, file_processor):
        """Test processing all Excel files in data/private folder."""
        data_dir = Path("data/private")
        excel_files = list(data_dir.glob("*.xlsx"))
        
        if not excel_files:
            pytest.skip("No Excel files found in data/private directory")
        
        results = {}
        
        for excel_file in excel_files:
            print(f"\nProcessing {excel_file.name}...")
            
            try:
                # Extract raw data
                raw_data = file_processor.process_file(excel_file)
                
                # Create workflow state
                workflow_state = WorkflowState(
                    workflow_id=f"test-{excel_file.stem}",
                    workflow_type="financial_analysis",
                    file_processing=FileProcessingState(
                        status=AgentStatus.COMPLETED,
                        raw_data=raw_data,
                        validation_errors=[],
                        processing_metadata={"source_file": str(excel_file)}
                    )
                )
                
                # Process with agent
                result_state = await agent.execute(workflow_state)
                analysis = result_state["analysis"]
                
                # Store results
                results[excel_file.name] = {
                    "raw_count": len(raw_data),
                    "processed_count": len(analysis["transactions"]),
                    "quality_score": float(analysis["metrics"]["quality_score"]),
                    "recovery_rate": float(analysis["metrics"]["recovery_rate"]),
                    "fraud_risk": float(analysis["metrics"]["fraud_risk_score"]),
                    "anomalies": len(analysis["anomalies"]),
                    "compliance_issues": len(analysis["compliance_issues"])
                }
                
                print(f"  Raw: {results[excel_file.name]['raw_count']}")
                print(f"  Processed: {results[excel_file.name]['processed_count']}")
                print(f"  Quality: {results[excel_file.name]['quality_score']:.2%}")
                print(f"  Recovery: {results[excel_file.name]['recovery_rate']:.2%}")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                results[excel_file.name] = {"error": str(e)}
        
        # Print summary
        print(f"\n=== PROCESSING SUMMARY ===")
        successful_files = [name for name, result in results.items() if "error" not in result]
        failed_files = [name for name, result in results.items() if "error" in result]
        
        print(f"Successfully processed: {len(successful_files)} files")
        print(f"Failed to process: {len(failed_files)} files")
        
        if successful_files:
            avg_quality = sum(results[name]["quality_score"] for name in successful_files) / len(successful_files)
            avg_recovery = sum(results[name]["recovery_rate"] for name in successful_files) / len(successful_files)
            total_processed = sum(results[name]["processed_count"] for name in successful_files)
            total_raw = sum(results[name]["raw_count"] for name in successful_files)
            
            print(f"Average Quality Score: {avg_quality:.2%}")
            print(f"Average Recovery Rate: {avg_recovery:.2%}")
            print(f"Total Transactions Processed: {total_processed} out of {total_raw}")
        
        # Verify at least some files were processed successfully
        assert len(successful_files) > 0, "No files were processed successfully"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_with_real_data(self, agent, file_processor):
        """Test anomaly detection capabilities with real data."""
        data_dir = Path("data/private")
        excel_files = list(data_dir.glob("*.xlsx"))
        
        if not excel_files:
            pytest.skip("No Excel files found in data/private directory")
        
        test_file = excel_files[0]
        
        # Process file
        raw_data = file_processor.process_file(test_file)
        workflow_state = WorkflowState(
            workflow_id="anomaly-test",
            workflow_type="financial_analysis",
            file_processing=FileProcessingState(
                status=AgentStatus.COMPLETED,
                raw_data=raw_data,
                validation_errors=[],
                processing_metadata={}
            )
        )
        
        result_state = await agent.execute(workflow_state)
        analysis = result_state["analysis"]
        
        # Test anomaly detection results
        anomalies = analysis["anomalies"]
        print(f"\nAnomaly Detection Results:")
        print(f"Total anomalies detected: {len(anomalies)}")
        
        # Group anomalies by type
        anomaly_types = {}
        for anomaly in anomalies:
            anomaly_type = anomaly.get("type", "unknown")
            if anomaly_type not in anomaly_types:
                anomaly_types[anomaly_type] = 0
            anomaly_types[anomaly_type] += 1
        
        for anomaly_type, count in anomaly_types.items():
            print(f"  {anomaly_type}: {count}")
        
        # Test fraud detection results
        compliance_issues = analysis["compliance_issues"]
        print(f"\nFraud Detection Results:")
        print(f"Total fraud indicators: {len(compliance_issues)}")
        
        for issue in compliance_issues:
            print(f"  - {issue}")
        
        # Verify anomaly detection is working
        # (We don't assert specific counts since real data varies)
        assert isinstance(anomalies, list), "Anomalies should be a list"
        assert isinstance(compliance_issues, list), "Compliance issues should be a list"
    
    @pytest.mark.asyncio
    async def test_data_recovery_with_real_data(self, agent, file_processor):
        """Test data recovery capabilities with real messy data."""
        data_dir = Path("data/private")
        excel_files = list(data_dir.glob("*.xlsx"))
        
        if not excel_files:
            pytest.skip("No Excel files found in data/private directory")
        
        test_file = excel_files[0]
        
        # Process file
        raw_data = file_processor.process_file(test_file)
        workflow_state = WorkflowState(
            workflow_id="recovery-test",
            workflow_type="financial_analysis",
            file_processing=FileProcessingState(
                status=AgentStatus.COMPLETED,
                raw_data=raw_data,
                validation_errors=[],
                processing_metadata={}
            )
        )
        
        result_state = await agent.execute(workflow_state)
        analysis = result_state["analysis"]
        
        # Check recovery report
        recovery_report = agent.get_memory("recovery_report", {})
        print(f"\nData Recovery Report:")
        print(f"Summary: {recovery_report.get('summary', 'No summary available')}")
        
        validation_results = recovery_report.get("validation_results", {})
        print(f"Passed validation: {validation_results.get('passed_validation', 0)}")
        print(f"Required recovery: {validation_results.get('required_recovery', 0)}")
        print(f"Failed processing: {validation_results.get('failed_processing', 0)}")
        
        # Check data quality improvements
        improvements = recovery_report.get("data_quality_improvements", [])
        print(f"Quality improvements applied:")
        for improvement in improvements:
            print(f"  - {improvement}")
        
        # Check recommendations
        recommendations = recovery_report.get("recommendations", [])
        if recommendations:
            print(f"Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        # Verify recovery capabilities are working
        processed_transactions = analysis["transactions"]
        recovery_rate = float(analysis["metrics"]["recovery_rate"])
        
        print(f"\nRecovery Statistics:")
        print(f"Raw records: {len(raw_data)}")
        print(f"Processed transactions: {len(processed_transactions)}")
        print(f"Recovery rate: {recovery_rate:.2%}")
        
        # Basic assertions
        assert recovery_rate >= 0, "Recovery rate should be non-negative"
        assert len(processed_transactions) <= len(raw_data), "Cannot process more than raw data"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])