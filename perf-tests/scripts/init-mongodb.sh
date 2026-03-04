#!/bin/sh
set -e

MONGO_URI="${MONGO_URI:-mongodb://localhost:27017/}"
DB_NAME="ai-defra-search-agent"


if [ -d "/docker-entrypoint-initdb.d/data" ]; then
  DATA_DIR="/docker-entrypoint-initdb.d/data"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  DATA_DIR="${SCRIPT_DIR}/data"
fi

if [ ! -f "${DATA_DIR}/knowledgeGroups.json" ]; then
  echo "ERROR: Required file not found: ${DATA_DIR}/knowledgeGroups.json"
  exit 1
fi

if [ ! -f "${DATA_DIR}/knowledgeSnapshots.json" ]; then
  echo "ERROR: Required file not found: ${DATA_DIR}/knowledgeSnapshots.json"
  exit 1
fi

echo "db.knowledgeGroups.drop(); db.knowledgeGroups.insertOne(EJSON.parse(cat('${DATA_DIR}/knowledgeGroups.json')));" | mongosh "${MONGO_URI%/}/${DB_NAME}"
echo "db.knowledgeSnapshots.drop(); db.knowledgeSnapshots.insertOne(EJSON.parse(cat('${DATA_DIR}/knowledgeSnapshots.json')));" | mongosh "${MONGO_URI%/}/${DB_NAME}"
