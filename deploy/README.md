# Serving

The service returns film recommendations for a user ID. It serves the two-tower arm the main
experiment scored (decision 037) and ranks films with the same function as the evaluation
(decision 036). Users the model never saw get the most rated films of its training window.

## API

| Route | Answer |
|---|---|
| `GET /recommendations/{user_id}?k=10` | `{"user_id": 1, "source": "model", "film_ids": [...]}`, where `source` is `model` or `popular` |
| `GET /healthz` | `{"status": "ok"}` once the bundle is loaded |
| `GET /metrics` | request counts and latencies in Prometheus format |

`k` runs from 1 to 100. Logs are one JSON object per line and record the route, never the user ID.

## Run it

The bundle is built from rating data, so it is never committed or copied into the image
(decision 038). Build it on Kaggle with `trs export-serving --config configs/serving.yaml`, or write
a synthetic one with a planted answer:

```
uv run python deploy/synthetic_bundle.py artifacts
```

Then, with Docker and [kind](https://kind.sigs.k8s.io/) installed, from the repository root:

```
deploy/cluster.sh          # three-node cluster, two copies of the service
deploy/smoke.sh 1 7        # from a pod inside the cluster: user 1 must get film 7 first
deploy/smoke.sh 999 5      # an unknown user must get film 5 first
kind delete cluster --name recsys
```

CI runs the same steps with the synthetic bundle on every pull request.

## Limits

The kind nodes are containers on one machine, so they share its CPUs and memory. The cluster tests
scheduling, health checks and networking between nodes, not the capacity of separate machines. The
service runs a fixed two copies without autoscaling (decision 039).
