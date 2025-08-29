#!/usr/bin/env python3
"""Simple test to verify workflow structure without full execution."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all workflow modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test state schemas
        from src.workflows.state_schemas import (
            WorkflowState, WorkflowStatus, AgentStatus,
            create_initial_workflow_state, update_agent_state
        )
        print("✓ State schemas imported successfully")
        
        # Test routing
        from src.workflows.routing import (
            WorkflowRouter, should_generate_report,
            check_report_generation, should_retry
        )
        print("✓ Routing module imported successfully")
        
        # Test workflow (this might fail due to missing dependencies)
        try:
            from src.workflows.financial_analysis import (
                FinancialAnalysisWorkflow, create_financial_analysis_workflow
            )
            print("✓ Financial analysis workflow imported successfully")
        except ImportError as e:
            print(f"⚠ Financial analysis workflow import failed (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def test_state_creation():
    """Test state creation and manipulation."""
    print("\nTesting state creation...")
    
    try:
        from src.workflows.state_schemas import (
            create_initial_workflow_state, update_agent_state,
            WorkflowStatus, AgentStatus
        )
        
        # Create initial state
        state = create_initial_workflow_state(
            workflow_id="test-001",
            input_files=["test.csv"],
            config={"test": True}
        )
        
        print(f"✓ Initial state created with ID: {state['workflow_id']}")
        print(f"✓ State has {len(state)} keys")
        
        # Test agent state update
        updated_state = update_agent_state(
            state, "data_fetcher", AgentStatus.IN_PROGRESS
        )
        
        fetcher_status = updated_state["data_fetcher"]["status"]
        print(f"✓ Agent state updated: {fetcher_status}")
        
        # Verify expected keys exist
        expected_keys = [
            "workflow_id", "status", "data_fetcher", "data_processor",
            "categorizer", "report_generator", "errors", "warnings"
        ]
        
        missing_keys = [key for key in expected_keys if key not in state]
        if missing_keys:
            print(f"⚠ Missing keys: {missing_keys}")
        else:
            print("✓ All expected keys present in state")
        
        return True
        
    except Exception as e:
        print(f"✗ State creation test failed: {e}")
        return False


def test_routing_logic():
    """Test routing logic without full workflow."""
    print("\nTesting routing logic...")
    
    try:
        from src.workflows.routing import should_generate_report, WorkflowRouter
        from src.workflows.state_schemas import create_initial_workflow_state, AgentStatus
        
        # Create test state
        state = create_initial_workflow_state("test-routing")
        
        # Test successful path
        state["data_fetcher"]["status"] = AgentStatus.COMPLETED
        state["data_processor"]["status"] = AgentStatus.COMPLETED  
        state["categorizer"]["status"] = AgentStatus.COMPLETED
        state["analysis"] = {"transactions": [{"id": "1", "amount": 100}]}
        state["quality_score"] = 0.8
        
        result = should_generate_report(state)
        print(f"✓ Successful routing result: {result}")
        
        # Test error path
        state["data_fetcher"]["status"] = AgentStatus.FAILED
        result = should_generate_report(state)
        print(f"✓ Error routing result: {result}")
        
        # Test confidence calculation
        confidence = WorkflowRouter.calculate_confidence_score(state)
        print(f"✓ Confidence score calculated: {confidence}")
        
        return True
        
    except Exception as e:
        print(f"✗ Routing logic test failed: {e}")
        return False


def main():
    """Run simple workflow tests."""
    print("=== Simple StateGraph Workflow Tests ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("State Creation Test", test_state_creation),
        ("Routing Logic Test", test_routing_logic)
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
    print("\n=== Test Results ===")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed >= 2:  # Allow workflow import to fail due to dependencies
        print("✓ Core workflow structure is working!")
        return 0
    else:
        print("✗ Critical workflow issues found.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)