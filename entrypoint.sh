#!/bin/sh
set -e

echo "==================================================================="
echo "Starting AI Defra Search Agent initialization..."
echo "==================================================================="

# Extract MongoDB connection details from MONGO_URI
MONGO_URI="${MONGO_URI:-mongodb://localhost:27017/}"
DB_NAME="ai-defra-search-agent"

MONGO_HOST=$(echo "$MONGO_URI" | sed 's|mongodb://||' | sed 's|/.*||' | cut -d: -f1)
MONGO_PORT=$(echo "$MONGO_URI" | sed 's|mongodb://||' | sed 's|/.*||' | cut -d: -f2)
MONGO_PORT=${MONGO_PORT:-27017}

echo "MongoDB Host: ${MONGO_HOST}"
echo "MongoDB Port: ${MONGO_PORT}"
echo "Database: ${DB_NAME}"

# Wait for MongoDB to be ready
echo "Waiting for MongoDB to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

until mongosh --host "${MONGO_HOST}" --port "${MONGO_PORT}" --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "ERROR: MongoDB is not available after ${MAX_RETRIES} attempts"
    exit 1
  fi
  echo "MongoDB not ready yet, retrying... (${RETRY_COUNT}/${MAX_RETRIES})"
  sleep 1
done

echo "✓ MongoDB is ready"

# Clear existing collections
echo "Clearing existing MongoDB collections..."
if ! mongosh --host "${MONGO_HOST}" --port "${MONGO_PORT}" --quiet /home/nonroot/perf-tests/scripts/clear-mongodb.js; then
  echo "ERROR: Failed to clear MongoDB collections"
  exit 1
fi
echo "✓ Collections cleared successfully"

# Initialize MongoDB with fresh data
echo "Initializing MongoDB with fresh data..."
if ! /home/nonroot/perf-tests/scripts/init-mongodb.sh; then
  echo "ERROR: Failed to initialize MongoDB"
  exit 1
fi

echo "==================================================================="
echo "MongoDB initialization completed. Starting application..."
echo "==================================================================="

# Start the application
exec /home/nonroot/.venv/bin/ai-defra-search-agent
