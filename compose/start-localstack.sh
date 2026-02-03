#!/bin/bash
set -e

export AWS_REGION=eu-west-2
export AWS_DEFAULT_REGION=eu-west-2
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test

echo "Waiting for LocalStack to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if awslocal sqs list-queues > /dev/null 2>&1; then
    echo "LocalStack is ready!"
    break
  fi
  attempt=$((attempt + 1))
  echo "Attempt $attempt/$max_attempts - LocalStack not ready yet..."
  sleep 1
done

if [ $attempt -eq $max_attempts ]; then
  echo "ERROR: LocalStack failed to become ready after $max_attempts attempts"
  exit 1
fi

echo "Creating SQS queue: ai-defra-search-agent-invoke"
awslocal sqs create-queue \
  --queue-name ai-defra-search-agent-invoke \
  --attributes VisibilityTimeout=300 \
  --region eu-west-2 2>&1 | grep -v "QueueAlreadyExists" || echo "Queue created or already exists"

echo "Verifying queue creation..."
queue_url=$(awslocal sqs get-queue-url --queue-name ai-defra-search-agent-invoke --region eu-west-2 --query 'QueueUrl' --output text)
echo "Queue URL: $queue_url"

echo "Listing all queues:"
awslocal sqs list-queues --region eu-west-2

echo "LocalStack initialization complete!"