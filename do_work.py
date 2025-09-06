import requests
import json
import time

print("Testing batch processing endpoint...")

try:
    # Process all files in uploads directory
    response = requests.post(
        'http://localhost:8000/api/v1/workflows/batch-uploads',
        json={
            'client_id': 'automated_batch',
            'workflow_type': 'financial_analysis'
        },
        timeout=10
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        workflow_id = result['workflow_id']
        files_processed = result.get('files_processed', 'N/A')
        print(f"✅ Batch processing started!")
        print(f"   Workflow ID: {workflow_id}")
        print(f"   Files processed: {files_processed}")

        # Monitor progress
        print("\n📊 Monitoring workflow progress...")
        for i in range(5):  # Check status 5 times
            try:
                status_response = requests.get(f'http://localhost:8000/api/v1/workflows/{workflow_id}', timeout=5)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data['status']} ({status_data.get('progress_percentage', 0):.1f}%)")
                    if status_data['status'] in ['completed', 'failed']:
                        break
                else:
                    print(f"   Status check failed: {status_response.status_code}")
            except Exception as e:
                print(f"   Status check error: {e}")

            if i < 4:  # Don't sleep after last check
                time.sleep(2)

    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"❌ Connection error: {e}")
    print("Make sure the FastAPI server is running on http://localhost:8000")
except Exception as e:
    print(f"❌ Unexpected error: {e}")