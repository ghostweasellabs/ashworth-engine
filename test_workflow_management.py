#!/usr/bin/env python3
"""
Test script for workflow management features.

This script tests all the workflow management endpoints including:
- Workflow creation with file uploads
- Status tracking and progress reporting
- Result retrieval with partial results
- Workflow cancellation and cleanup
- Workflow interruption and resumption
- Health checks for all services
- API documentation validation
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import httpx
import pandas as pd


class WorkflowManagementTester:
    """Test suite for workflow management features."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
    
    async def test_health_checks(self):
        """Test health check endpoints."""
        print("\n🔍 Testing Health Check Endpoints")
        
        # Test basic health check
        try:
            response = await self.client.get(f"{self.base_url}/health")
            success = response.status_code in [200, 207, 503]  # Any of these is acceptable
            data = response.json()
            
            self.log_test(
                "Basic Health Check",
                success,
                f"Status: {response.status_code}, Health: {data.get('status', 'unknown')}"
            )
        except Exception as e:
            self.log_test("Basic Health Check", False, f"Error: {str(e)}")
        
        # Test detailed health check
        try:
            response = await self.client.get(f"{self.base_url}/health/detailed")
            success = response.status_code in [200, 503]
            data = response.json()
            
            services = data.get("services", {})
            healthy_services = sum(1 for status in services.values() 
                                 if status.get("status") == "healthy")
            
            self.log_test(
                "Detailed Health Check",
                success,
                f"Services: {len(services)}, Healthy: {healthy_services}"
            )
        except Exception as e:
            self.log_test("Detailed Health Check", False, f"Error: {str(e)}")
    
    def create_test_files(self) -> list:
        """Create test files for upload."""
        test_files = []
        
        # Create a test CSV file
        csv_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'amount': [100.50, -25.00, 75.25],
            'description': ['Revenue', 'Office Supplies', 'Consulting Fee'],
            'category': ['Income', 'Expense', 'Income']
        })
        
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        csv_data.to_csv(csv_file.name, index=False)
        test_files.append(csv_file.name)
        
        # Create a test Excel file
        excel_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        csv_data.to_excel(excel_file.name, index=False)
        test_files.append(excel_file.name)
        
        return test_files
    
    async def test_workflow_creation(self) -> str:
        """Test workflow creation with file uploads."""
        print("\n📁 Testing Workflow Creation")
        
        test_files = self.create_test_files()
        workflow_id = None
        
        try:
            # Prepare files for upload
            files = []
            for file_path in test_files:
                with open(file_path, 'rb') as f:
                    files.append(('files', (Path(file_path).name, f.read(), 'application/octet-stream')))
            
            # Create workflow
            data = {
                'client_id': 'test_client_123',
                'workflow_type': 'financial_analysis'
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/workflows",
                files=files,
                data=data
            )
            
            success = response.status_code == 200
            if success:
                result = response.json()
                workflow_id = result.get('workflow_id')
                
                self.log_test(
                    "Workflow Creation",
                    True,
                    f"Created workflow: {workflow_id}, Status: {result.get('status')}"
                )
            else:
                self.log_test(
                    "Workflow Creation",
                    False,
                    f"Status: {response.status_code}, Response: {response.text}"
                )
        
        except Exception as e:
            self.log_test("Workflow Creation", False, f"Error: {str(e)}")
        
        finally:
            # Clean up test files
            for file_path in test_files:
                try:
                    Path(file_path).unlink()
                except:
                    pass
        
        return workflow_id
    
    async def test_workflow_status_tracking(self, workflow_id: str):
        """Test workflow status tracking and progress reporting."""
        print("\n📊 Testing Workflow Status Tracking")
        
        if not workflow_id:
            self.log_test("Workflow Status Tracking", False, "No workflow ID provided")
            return
        
        try:
            # Test status endpoint
            response = await self.client.get(f"{self.base_url}/api/v1/workflows/{workflow_id}")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                agents = data.get('agents', [])
                progress = data.get('progress_percentage', 0)
                
                self.log_test(
                    "Workflow Status Tracking",
                    True,
                    f"Status: {data.get('status')}, Progress: {progress}%, Agents: {len(agents)}"
                )
                
                # Test agent-level status
                agent_statuses = [agent.get('status') for agent in agents]
                unique_statuses = set(agent_statuses)
                
                self.log_test(
                    "Agent-Level Status Tracking",
                    len(agents) > 0,
                    f"Agent statuses: {list(unique_statuses)}"
                )
            else:
                self.log_test(
                    "Workflow Status Tracking",
                    False,
                    f"Status: {response.status_code}, Response: {response.text}"
                )
        
        except Exception as e:
            self.log_test("Workflow Status Tracking", False, f"Error: {str(e)}")
    
    async def test_partial_results(self, workflow_id: str):
        """Test result retrieval with partial results."""
        print("\n📋 Testing Partial Results Retrieval")
        
        if not workflow_id:
            self.log_test("Partial Results Retrieval", False, "No workflow ID provided")
            return
        
        try:
            # Test partial results while workflow is running
            response = await self.client.get(
                f"{self.base_url}/api/v1/workflows/{workflow_id}/results?include_partial=true"
            )
            
            success = response.status_code in [200, 202]
            
            if success:
                data = response.json()
                processing_summary = data.get('processing_summary', {})
                is_partial = processing_summary.get('is_partial', False)
                completed_agents = processing_summary.get('completed_agents', [])
                
                self.log_test(
                    "Partial Results Retrieval",
                    True,
                    f"Partial: {is_partial}, Completed agents: {len(completed_agents)}"
                )
            else:
                self.log_test(
                    "Partial Results Retrieval",
                    False,
                    f"Status: {response.status_code}, Response: {response.text}"
                )
        
        except Exception as e:
            self.log_test("Partial Results Retrieval", False, f"Error: {str(e)}")
    
    async def test_workflow_interruption(self, workflow_id: str):
        """Test workflow interruption capabilities."""
        print("\n⏸️ Testing Workflow Interruption")
        
        if not workflow_id:
            self.log_test("Workflow Interruption", False, "No workflow ID provided")
            return
        
        try:
            # Test interrupt request
            interrupt_data = {
                "interrupt_before": "report_generator",
                "message": "Test interruption for review"
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/workflows/{workflow_id}/interrupt",
                json=interrupt_data
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                self.log_test(
                    "Workflow Interruption",
                    True,
                    f"Interrupt status: {data.get('status')}, Before: {data.get('interrupt_before')}"
                )
                
                # Test resume
                await asyncio.sleep(1)  # Brief pause
                
                resume_data = {
                    "message": "Test resumption after review"
                }
                
                resume_response = await self.client.post(
                    f"{self.base_url}/api/v1/workflows/{workflow_id}/resume",
                    json=resume_data
                )
                
                resume_success = resume_response.status_code == 200
                if resume_success:
                    resume_result = resume_response.json()
                    resolved_count = resume_result.get('resolved_interrupts', 0)
                    
                    self.log_test(
                        "Workflow Resumption",
                        True,
                        f"Resolved interrupts: {resolved_count}"
                    )
                else:
                    self.log_test(
                        "Workflow Resumption",
                        False,
                        f"Status: {resume_response.status_code}"
                    )
            else:
                self.log_test(
                    "Workflow Interruption",
                    False,
                    f"Status: {response.status_code}, Response: {response.text}"
                )
        
        except Exception as e:
            self.log_test("Workflow Interruption", False, f"Error: {str(e)}")
    
    async def test_workflow_cancellation(self, workflow_id: str):
        """Test workflow cancellation and cleanup."""
        print("\n🛑 Testing Workflow Cancellation")
        
        if not workflow_id:
            self.log_test("Workflow Cancellation", False, "No workflow ID provided")
            return
        
        try:
            # Wait a moment to ensure workflow is running
            await asyncio.sleep(2)
            
            response = await self.client.delete(f"{self.base_url}/api/v1/workflows/{workflow_id}")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                cancelled_agents = data.get('cancelled_agents', [])
                cleaned_files = data.get('cleaned_files', 0)
                
                self.log_test(
                    "Workflow Cancellation",
                    True,
                    f"Cancelled agents: {len(cancelled_agents)}, Cleaned files: {cleaned_files}"
                )
                
                # Verify workflow status is cancelled
                await asyncio.sleep(1)
                status_response = await self.client.get(f"{self.base_url}/api/v1/workflows/{workflow_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    final_status = status_data.get('status')
                    
                    self.log_test(
                        "Cancellation Status Verification",
                        final_status == "cancelled",
                        f"Final status: {final_status}"
                    )
            else:
                self.log_test(
                    "Workflow Cancellation",
                    False,
                    f"Status: {response.status_code}, Response: {response.text}"
                )
        
        except Exception as e:
            self.log_test("Workflow Cancellation", False, f"Error: {str(e)}")
    
    async def test_workflow_listing(self):
        """Test workflow listing with filtering and pagination."""
        print("\n📋 Testing Workflow Listing")
        
        try:
            # Test basic listing
            response = await self.client.get(f"{self.base_url}/api/v1/workflows")
            success = response.status_code == 200
            
            if success:
                data = response.json()
                workflows = data.get('workflows', [])
                total = data.get('total', 0)
                
                self.log_test(
                    "Workflow Listing",
                    True,
                    f"Total workflows: {total}, Returned: {len(workflows)}"
                )
                
                # Test filtering
                filter_response = await self.client.get(
                    f"{self.base_url}/api/v1/workflows?client_id=test_client_123&limit=10"
                )
                
                filter_success = filter_response.status_code == 200
                if filter_success:
                    filter_data = filter_response.json()
                    filtered_workflows = filter_data.get('workflows', [])
                    
                    self.log_test(
                        "Workflow Filtering",
                        True,
                        f"Filtered workflows: {len(filtered_workflows)}"
                    )
            else:
                self.log_test(
                    "Workflow Listing",
                    False,
                    f"Status: {response.status_code}, Response: {response.text}"
                )
        
        except Exception as e:
            self.log_test("Workflow Listing", False, f"Error: {str(e)}")
    
    async def test_api_documentation(self):
        """Test API documentation endpoints."""
        print("\n📚 Testing API Documentation")
        
        try:
            # Test OpenAPI schema
            response = await self.client.get(f"{self.base_url}/openapi.json")
            success = response.status_code == 200
            
            if success:
                schema = response.json()
                paths = schema.get('paths', {})
                components = schema.get('components', {})
                
                # Check for key endpoints
                key_endpoints = [
                    '/api/v1/workflows',
                    '/api/v1/workflows/{workflow_id}',
                    '/api/v1/workflows/{workflow_id}/results',
                    '/health'
                ]
                
                found_endpoints = sum(1 for endpoint in key_endpoints if endpoint in paths)
                
                self.log_test(
                    "OpenAPI Schema",
                    found_endpoints == len(key_endpoints),
                    f"Found {found_endpoints}/{len(key_endpoints)} key endpoints"
                )
                
                # Check for file upload examples
                workflow_post = paths.get('/api/v1/workflows', {}).get('post', {})
                has_file_upload = 'multipart/form-data' in str(workflow_post)
                
                self.log_test(
                    "File Upload Documentation",
                    has_file_upload,
                    "File upload examples in OpenAPI schema"
                )
            else:
                self.log_test(
                    "OpenAPI Schema",
                    False,
                    f"Status: {response.status_code}"
                )
            
            # Test Swagger UI
            docs_response = await self.client.get(f"{self.base_url}/docs")
            docs_success = docs_response.status_code == 200
            
            self.log_test(
                "Swagger UI Accessibility",
                docs_success,
                "Swagger documentation accessible"
            )
        
        except Exception as e:
            self.log_test("API Documentation", False, f"Error: {str(e)}")
    
    async def run_all_tests(self):
        """Run all workflow management tests."""
        print("🚀 Starting Workflow Management Feature Tests")
        print("=" * 60)
        
        # Test health checks first
        await self.test_health_checks()
        
        # Test API documentation
        await self.test_api_documentation()
        
        # Create a workflow for testing
        workflow_id = await self.test_workflow_creation()
        
        if workflow_id:
            # Test status tracking
            await self.test_workflow_status_tracking(workflow_id)
            
            # Test partial results
            await self.test_partial_results(workflow_id)
            
            # Test interruption
            await self.test_workflow_interruption(workflow_id)
            
            # Test cancellation (this will end the workflow)
            await self.test_workflow_cancellation(workflow_id)
        
        # Test workflow listing
        await self.test_workflow_listing()
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['details']}")
        
        print("\n🎉 All workflow management features tested!")


async def main():
    """Main test execution."""
    async with WorkflowManagementTester() as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())