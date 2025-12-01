#!/usr/bin/env bash

set -e

current_date=$(date -u +%s)
expired_found=false

while IFS= read -r line; do
  if [[ "$line" =~ ^#[[:space:]]*Expires:[[:space:]]*(.+)$ ]]; then
    expires_date="${BASH_REMATCH[1]}"
    expires_timestamp=$(date -u -d "$expires_date" +%s 2>/dev/null || echo 0)
    
    if [ "$expires_timestamp" -ne 0 ] && [ "$expires_timestamp" -lt "$current_date" ]; then
      echo "ERROR: Expired Trivy ignore found: $expires_date"
      expired_found=true
    fi
  fi
done < .trivyignore

if [ "$expired_found" = true ]; then
  echo "One or more Trivy ignores have expired. Please review and update .trivyignore"
  exit 1
fi

echo "All Trivy ignores are valid"
