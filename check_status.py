#!/usr/bin/env python3
"""
Check workflow status
"""

import requests

def check_status():
    workflow_id = '607689d8-bb52-464d-84c1-1300d549118a'
    response = requests.get(f'http://localhost:8000/api/v1/workflows/{workflow_id}')

    if response.status_code == 200:
        status = response.json()
        print('=== WORKFLOW STATUS ===')
        print(f'Status: {status["status"]}')
        print(f'Progress: {status.get("progress_percentage", 0)}%')

        agents = status.get('agents', [])
        print(f'\nAgents found: {len(agents)}')
        for agent in agents:
            print(f'- {agent.get("agent_name", "Unknown")}: {agent.get("status", "unknown")} ({agent.get("duration_ms", 0)}ms)')

        errors = status.get('errors', [])
        if errors:
            print(f'\nErrors: {len(errors)}')
            for error in errors:
                print(f'- {error}')

        input_files = status.get('input_files', [])
        pdf_files = [f for f in input_files if f.lower().endswith('.pdf')]
        print(f'\nPDF files: {len(pdf_files)}')
        if pdf_files:
            print('Dr. Sharpe should have run on these files')
    else:
        print(f'Error getting status: {response.status_code}')
        print(response.text)

if __name__ == "__main__":
    check_status()
