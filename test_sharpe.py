#!/usr/bin/env python3
"""
Test Dr. Evelyn Sharpe with Gemma3 vision
"""

import requests
import json
import time

def test_sharpe():
    print('🚀 Testing Dr. Evelyn Sharpe with Gemma3 vision...')

    # Test batch processing
    response = requests.post(
        'http://localhost:8000/api/v1/workflows/batch-uploads',
        json={'client_id': 'gemma3_final_test', 'workflow_type': 'financial_analysis'}
    )

    print(f'Status: {response.status_code}')
    if response.status_code in [200, 202]:
        result = response.json()
        workflow_id = result['workflow_id']
        print(f'✅ Workflow created: {workflow_id}')
        print(f'📄 Files to process: {result.get("files_processed", "unknown")}')

        # Monitor progress
        print('\n📊 Monitoring progress...')
        for i in range(10):  # Check 10 times
            try:
                status_response = requests.get(f'http://localhost:8000/api/v1/workflows/{workflow_id}')
                if status_response.status_code == 200:
                    status = status_response.json()
                    progress = status.get('progress_percentage', 0)
                    current_agent = status.get('current_agent', 'unknown')

                    print(f'   {i+1}/10 - Status: {status["status"]} ({progress}%) - Agent: {current_agent}')

                    # Check agent status
                    agents = status.get('agents', [])
                    sharpe_status = next((a for a in agents if 'sharpe' in a.get('agent_name', '').lower()), None)
                    if sharpe_status:
                        print(f'      🕵️‍♀️ Dr. Sharpe: {sharpe_status["status"]} ({sharpe_status.get("duration_ms", 0)}ms)')

                    if status['status'] in ['completed', 'failed']:
                        print(f'\n🎯 Final Status: {status["status"]}')
                        break
                else:
                    print(f'   Status check failed: {status_response.status_code}')
            except Exception as e:
                print(f'   Error checking status: {e}')

            if i < 9:  # Don't sleep on last iteration
                time.sleep(2)

        # Get final results
        try:
            results_response = requests.get(f'http://localhost:8000/api/v1/workflows/{workflow_id}/results')
            if results_response.status_code == 200:
                results = results_response.json()
                print(f'\n📊 Final Results:')
                print(f'   Status: {results["status"]}')
                print(f'   Transactions Found: {results["processing_summary"]["total_transactions"]}')
                print(f'   Files Processed: {results["processing_summary"]["total_files_processed"]}')

                if results["processing_summary"]["total_transactions"] > 0:
                    print('   🎉 SUCCESS! Dr. Sharpe extracted financial data!')
                else:
                    print('   🔍 Dr. Sharpe completed analysis but found no transactions')
            else:
                print(f'   Error getting results: {results_response.status_code}')
        except Exception as e:
            print(f'   Error getting final results: {e}')

    else:
        print(f'❌ Error creating workflow: {response.status_code}')
        print(f'Response: {response.text}')

if __name__ == "__main__":
    test_sharpe()

