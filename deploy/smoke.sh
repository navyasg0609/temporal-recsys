#!/usr/bin/env bash
# Ask the service for recommendations from a pod inside the cluster.
# Usage: deploy/smoke.sh USER_ID [EXPECTED_FIRST_FILM]
set -euo pipefail

user_id=$1
expected=${2:-}
response=$(kubectl run "smoke-$RANDOM" --rm -i --restart=Never --quiet \
  --image=curlimages/curl:8.11.1 -- curl -sf "http://recsys/recommendations/$user_id?k=5")
echo "$response"
if [[ -n $expected && $response != *"\"film_ids\":[$expected,"* ]]; then
  echo "expected film $expected first for user $user_id" >&2
  exit 1
fi
