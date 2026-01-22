#!/usr/bin/env python3
"""Simple test script to verify SSE chat endpoint"""
import requests
import json

def test_chat():
    url = "http://localhost:8086/chat"
    payload = {
        "question": "What is AI in one sentence?",
        "modelId": "anthropic.claude-3-haiku"
    }
    
    print(f"Sending request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("\n--- SSE Stream Events ---\n")
    
    with requests.post(url, json=payload, stream=True, timeout=30) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        
        event_count = 0
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(decoded_line)
                
                # Parse and show structured data
                if decoded_line.startswith('data:'):
                    try:
                        data = json.loads(decoded_line[5:].strip())
                        event_count += 1
                        print(f"  → Event #{event_count}: status={data.get('status')}")
                        if data.get('status') in ['completed', 'failed']:
                            print(f"\n✓ Terminal status received, closing stream")
                            break
                    except json.JSONDecodeError:
                        pass

if __name__ == "__main__":
    test_chat()
