#!/bin/sh
set -e

echo "==================================================================="
echo "Starting AI Defra Search Agent initialization..."
echo "==================================================================="

# Seed MongoDB with perf-test data
echo "Seeding MongoDB..."
if ! /home/nonroot/.venv/bin/python /home/nonroot/perf-tests/scripts/seed-mongodb.py; then
  echo "ERROR: Failed to seed MongoDB"
  exit 1
fi
echo "✓ MongoDB seeded successfully"

echo "==================================================================="
echo "MongoDB initialization completed. Starting application..."
echo "==================================================================="

# Start the application
exec /home/nonroot/.venv/bin/ai-defra-search-agent
