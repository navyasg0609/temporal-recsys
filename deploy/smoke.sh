#!/usr/bin/env bash
# Ask the service for recommendations from a pod inside the cluster.
# Usage: deploy/smoke.sh USER_ID [EXPECTED_FIRST_FILM]
set -euo pipefail

user_id=$1
expected=${2:-}
pod="smoke-$RANDOM"
# Wait for the pod and read its log, because attaching to a pod that exits quickly can
# lose its output.
kubectl run "$pod" --restart=Never --image=curlimages/curl:8.11.1 \
  -- curl -sf "http://recsys/recommendations/$user_id?k=5" >/dev/null
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$pod" --timeout=60s >/dev/null
response=$(kubectl logs "$pod")
kubectl delete pod "$pod" --wait=false >/dev/null
echo "$response"
if [[ -n $expected && $response != *"\"film_ids\":[$expected,"* ]]; then
  echo "expected film $expected first for user $user_id" >&2
  exit 1
fi
