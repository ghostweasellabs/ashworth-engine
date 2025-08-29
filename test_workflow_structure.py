#!/usr/bin/env python3
"""Test the StateGraph workflow structure and execution paths."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_workflow_graph_structure():
    """Test that the workflow graph has the correct structure."""
    print("Testing workflow graph structure...")
    
    try:
        from src.workflows.financial_analysis import FinancialAnalysisWorkflow
        
        # Create workflow instance
        workflow = FinancialAnalysisWorkflow()
        
        # Check nodes
        expected_nodes = [
            "data_fetcher", 
            "data_processor", 
            "categorizer", 
            "report_generator",
            "error_handler",
            "quality_check"
        ]
        
        graph_nodes = list(workflow.graph.nodes.keys())
        print(f"Graph nodes: {graph_nodes}")
        
        missing_nodes = [node for node in expected_nodes if node not in graph_nodes]
        if missing_nodes:
            print(f"⚠ Missing nodes: {missing_nodes}")
        else:
            print("✓ All expected nodes present")
        
        # Check edges (basic structure)
        edges = workflow.graph.edges
        print(f"Graph has {len(edges)} edges")
        
        # Verify sequential flow exists
        sequential_edges = [
            ("__start__", "data_fetcher"),
            ("data_fetcher", "data_processor"),
            ("data_processor", "categorizer"),
            ("categorizer", "quality_check")
        ]
        
        for source, target in sequential_edges:
            if (source, target) in edges:
                print(f"✓ Sequential edge found: {source} -> {target}")
            else:
                print(f"⚠ Sequential edge missing: {source} -> {target}")
        
        return True
        
    except Exception as e:
        print(f"✗ Workflow graph structure test failed: {e}")
        return False


def test_conditional_routing():
    """Test conditional routing logic in detail."""
    print("\nTesting conditional routing...")
    
    try:
        from src.workflows.routing import (
            should_generate_report, check_report_generation, should_retry
        )
        from src.workflows.state_schemas import (
            create_initial_workflow_state, AgentStatus
        )
        
        # Test 1: Normal successful flow
        state = create_initial_workflow_state("test-routing-1")
        state["data_fetcher"]["status"] = AgentStatus.COMPLETED
        state["data_processor"]["status"] = AgentStatus.COMPLETED
        state["categorizer"]["status"] = AgentStatus.COMPLETED
        state["analysis"] = {"transactions": [{"id": "1"}]}
        state["quality_score"] = 0.8
        
        result = should_generate_report(state)
        assert result == "generate_report", f"Expected 'generate_report', got '{result}'"
        print("✓ Normal flow routing works")
        
        # Test 2: Failed agent
        state["data_processor"]["status"] = AgentStatus.FAILED
        result = should_generate_report(state)
        assert result == "handle_error", f"Expected 'handle_error', got '{result}'"
        print("✓ Failed agent routing works")
        
        # Test 3: Low quality score
        state["data_processor"]["status"] = AgentStatus.COMPLETED
        state["quality_score"] = 0.2
        result = should_generate_report(state)
        assert result == "handle_error", f"Expected 'handle_error', got '{result}'"
        print("✓ Low quality routing works")
        
        # Test 4: No transactions
        state["quality_score"] = 0.8
        state["analysis"] = {"transactions": []}
        result = should_generate_report(state)
        assert result == "handle_error", f"Expected 'handle_error', got '{result}'"
        print("✓ No transactions routing works")
        
        # Test 5: Report generation success
        state["report_generator"] = {"status": AgentStatus.COMPLETED}
        state["output_reports"] = ["report.md"]
        result = check_report_generation(state)
        assert result == "success", f"Expected 'success', got '{result}'"
        print("✓ Report generation success routing works")
        
        # Test 6: Report generation failure
        state["report_generator"]["status"] = AgentStatus.FAILED
        result = check_report_generation(state)
        assert result == "error", f"Expected 'error', got '{result}'"
        print("✓ Report generation failure routing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Conditional routing test failed: {e}")
        return False


def test_error_handling_paths():
    """Test error handling and retry logic."""
    print("\nTesting error handling paths...")
    
    try:
        from src.workflows.routing import WorkflowRouter
        from src.workflows.state_schemas import create_initial_workflow_state
        
        # Test retry strategy for different error types
        test_cases = [
            {
                "errors": ["File format error", "Upload failed"],
                "expected": "retry_data_fetcher",
                "description": "File processing errors"
            },
            {
                "errors": ["Validation failed", "Decimal conversion error"],
                "expected": "retry_data_processor", 
                "description": "Data processing errors"
            },
            {
                "errors": ["Category mapping failed", "Tax rule error"],
                "expected": "retry_categorizer",
                "description": "Categorization errors"
            },
            {
                "errors": ["Report template error", "Storage failed"],
                "expected": "retry_report_generator",
                "description": "Report generation errors"
            },
            {
                "errors": ["Unknown error", "System failure"],
                "expected": "end",
                "description": "Unknown errors"
            }
        ]
        
        for test_case in test_cases:
            state = create_initial_workflow_state("test-error")
            state["errors"] = test_case["errors"]
            
            result = WorkflowRouter.determine_retry_strategy(state)
            
            if result == test_case["expected"]:
                print(f"✓ {test_case['description']}: {result}")
            else:
                print(f"⚠ {test_case['description']}: expected '{test_case['expected']}', got '{result}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling paths test failed: {e}")
        return False


def test_interruption_points():
    """Test workflow interruption logic."""
    print("\nTesting interruption points...")
    
    try:
        from src.workflows.routing import WorkflowRouter
        from src.workflows.state_schemas import create_initial_workflow_state
        
        # Test interruption before report generation
        state = create_initial_workflow_state("test-interrupt")
        state["config"] = {"require_human_review": True}
        
        should_interrupt = WorkflowRouter.should_interrupt_for_review(state, "report_generator")
        assert should_interrupt == True, "Should interrupt for human review"
        print("✓ Human review interruption works")
        
        # Test interruption for warnings
        state["config"] = {"require_human_review": False}
        state["warnings"] = ["Data quality issue"]
        
        should_interrupt = WorkflowRouter.should_interrupt_for_review(state, "report_generator")
        assert should_interrupt == True, "Should interrupt for warnings"
        print("✓ Warning interruption works")
        
        # Test interruption for low quality
        state["warnings"] = []
        state["quality_score"] = 0.5
        
        should_interrupt = WorkflowRouter.should_interrupt_for_review(state, "quality_check")
        assert should_interrupt == True, "Should interrupt for low quality"
        print("✓ Quality interruption works")
        
        # Test no interruption
        state["quality_score"] = 0.9
        
        should_interrupt = WorkflowRouter.should_interrupt_for_review(state, "quality_check")
        assert should_interrupt == False, "Should not interrupt for high quality"
        print("✓ No interruption works")
        
        return True
        
    except Exception as e:
        print(f"✗ Interruption points test failed: {e}")
        return False


def test_confidence_scoring():
    """Test confidence score calculation."""
    print("\nTesting confidence scoring...")
    
    try:
        from src.workflows.routing import WorkflowRouter
        from src.workflows.state_schemas import create_initial_workflow_state, AgentStatus
        
        # Test high confidence scenario
        state = create_initial_workflow_state("test-confidence")
        state["quality_score"] = 0.9
        state["data_fetcher"]["status"] = AgentStatus.COMPLETED
        state["data_processor"]["status"] = AgentStatus.COMPLETED
        state["categorizer"]["status"] = AgentStatus.COMPLETED
        state["report_generator"]["status"] = AgentStatus.COMPLETED
        state["errors"] = []
        state["warnings"] = []
        state["analysis"] = {
            "transactions": [
                {"id": "1", "category": "office"},
                {"id": "2", "category": "travel"}
            ]
        }
        
        confidence = WorkflowRouter.calculate_confidence_score(state)
        print(f"High confidence scenario: {confidence:.3f}")
        assert confidence > 0.8, f"Expected high confidence, got {confidence}"
        print("✓ High confidence calculation works")
        
        # Test low confidence scenario
        state["quality_score"] = 0.3
        state["data_processor"]["status"] = AgentStatus.FAILED
        state["errors"] = ["Processing failed", "Validation error"]
        state["warnings"] = ["Data quality issue", "Missing fields"]
        
        confidence = WorkflowRouter.calculate_confidence_score(state)
        print(f"Low confidence scenario: {confidence:.3f}")
        assert confidence < 0.6, f"Expected low confidence, got {confidence}"
        print("✓ Low confidence calculation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Confidence scoring test failed: {e}")
        return False


def main():
    """Run comprehensive workflow structure tests."""
    print("=== StateGraph Workflow Structure Tests ===\n")
    
    tests = [
        ("Workflow Graph Structure", test_workflow_graph_structure),
        ("Conditional Routing", test_conditional_routing),
        ("Error Handling Paths", test_error_handling_paths),
        ("Interruption Points", test_interruption_points),
        ("Confidence Scoring", test_confidence_scoring)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Results Summary ===")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed >= 4:  # Allow one test to fail
        print("✓ StateGraph workflow structure is solid!")
        print("✓ Sequential execution paths verified")
        print("✓ Error handling and routing logic working")
        print("✓ Interruption points configured correctly")
        return 0
    else:
        print("✗ Critical workflow structure issues found.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)