#!/usr/bin/env python3
"""Simple test script to verify async chat endpoint with polling"""

import json
import os
import time

import pytest
import requests

# This file is an integration-style script that exercises a running local
# service. Skip it by default in automated CI unless explicitly enabled.
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_CHAT_INTEGRATION", "") != "true",
    reason="Integration test against a running local service",
)


def test_chat():
    url = "http://localhost:8086/chat"
    payload = {
        "question": "What is AI in one sentence?",
        "modelId": "anthropic.claude-3-haiku",
    }

    print(f"Sending request to {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    # Step 1: Queue the message
    response = requests.post(url, json=payload, timeout=10)

    if response.status_code != 202:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    result = response.json()
    conversation_id = result["conversation_id"]
    message_id = result["message_id"]
    status = result["status"]

    print("\n✓ Message queued successfully")
    print(f"  Conversation ID: {conversation_id}")
    print(f"  Message ID: {message_id}")
    print(f"  Initial Status: {status}")

    # Step 2: Poll for results
    print("\n--- Polling for Results ---\n")
    conversation_url = f"http://localhost:8086/conversations/{conversation_id}"

    max_attempts = 30
    poll_interval = 2  # seconds

    for attempt in range(1, max_attempts + 1):
        time.sleep(poll_interval)

        conv_response = requests.get(conversation_url, timeout=10)
        if conv_response.status_code != 200:
            print(f"Error fetching conversation: {conv_response.status_code}")
            continue

        conversation = conv_response.json()
        messages = conversation.get("messages", [])

        if not messages:
            continue

        # Find the user message we sent
        user_message = next(
            (msg for msg in messages if msg["message_id"] == message_id), None
        )

        if user_message:
            current_status = user_message["status"]
            print(
                f"Attempt {attempt}/{max_attempts}: Status = {current_status}", end=""
            )

            if current_status == "completed":
                print("\n\n✓ Message processing completed!")
                print(f"\nConversation ({len(messages)} messages):")
                for i, msg in enumerate(messages, 1):
                    print(
                        f"\n  Message {i} ({msg['role']}) - {msg.get('model_name', 'N/A')}"
                    )
                    print(f"  Status: {msg['status']}")
                    content_preview = (
                        msg["content"][:100] + "..."
                        if len(msg["content"]) > 100
                        else msg["content"]
                    )
                    print(f"  Content: {content_preview}")
                return

            if current_status == "failed":
                print("\n\n✗ Message processing failed!")
                print(f"  Error: {user_message.get('error_message', 'Unknown error')}")
                print(f"  Error Code: {user_message.get('error_code', 'N/A')}")
                return

            print()  # New line for next attempt

    print(
        f"\n✗ Timeout: Message did not complete within {max_attempts * poll_interval} seconds"
    )


if __name__ == "__main__":
    test_chat()
