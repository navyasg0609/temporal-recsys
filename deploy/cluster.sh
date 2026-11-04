#!/usr/bin/env bash
# Create the kind cluster if needed, build and load the image, and deploy the service.
# Run from the repository root, with a bundle at artifacts/bundle.npz.
set -euo pipefail

if [[ ! -f artifacts/bundle.npz ]]; then
  echo "artifacts/bundle.npz is missing" >&2
  exit 1
fi
if ! kind get clusters | grep -qx recsys; then
  kind create cluster --config deploy/kind.yaml
fi
docker build -f deploy/Dockerfile -t temporal-recsys:dev .
kind load docker-image temporal-recsys:dev --name recsys
kubectl apply -f deploy/k8s/service.yaml
# The tag does not change between builds, so restart to pick up a new image.
kubectl rollout restart deployment/recsys
kubectl rollout status deployment/recsys --timeout=180s
kubectl get pods -l app=recsys -o wide
