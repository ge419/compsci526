#!/bin/bash
# Test the dietary restriction API endpoint

CONTROLLER_URL="http://localhost:8000"

echo "Testing /health endpoint..."
curl -s "${CONTROLLER_URL}/health" | python3 -m json.tool

echo -e "\n\nTesting /set-dietary-restriction endpoint..."
curl -s -X POST "${CONTROLLER_URL}/set-dietary-restriction" \
  -H "Content-Type: application/json" \
  -d '{"restriction": "vegan"}' | python3 -m json.tool

echo -e "\n\nTesting /health again to verify restriction was set..."
curl -s "${CONTROLLER_URL}/health" | python3 -m json.tool
