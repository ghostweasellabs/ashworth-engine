#!/usr/bin/env python3
"""
Test Gemma3 vision API directly
"""

import asyncio
import httpx

async def test_gemma3_vision():
    # Test the Gemma3 vision API directly with a simple image
    payload = {
        'model': 'gemma3:12b',
        'prompt': 'Describe what you see in this image in detail.',
        'images': ['iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='],  # 1x1 red pixel
        'stream': False
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print("Testing Gemma3 vision API...")
            response = await client.post('http://192.168.1.220:11434/api/generate', json=payload)
            print(f'Status: {response.status_code}')
            if response.status_code == 200:
                result = response.json()
                print(f'Response: {result.get("response", "No response")}')
                print("✅ Gemma3 vision is working!")
            else:
                print(f'Error: {response.text}')
                print("❌ Gemma3 vision failed")
        except Exception as e:
            print(f'Exception: {e}')
            print("❌ Connection failed")

if __name__ == "__main__":
    asyncio.run(test_gemma3_vision())
