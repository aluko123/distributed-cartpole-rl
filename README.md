# Distributed CartPole RL (MVP Skeleton)

This repo is intentionally minimal. It defines the project layout and interfaces before implementation.

## Components

- `cmd/rollout-worker/`: Go worker processes that generate trajectories and push them to the buffer.
- `cmd/replay-buffer/`: Go replay buffer service with bounded queue and policy options.
- `trainer/`: Python PPO trainer that pulls batches and publishes policies.
- `metrics/`: Prometheus and Grafana configuration (later).
- `scripts/`: Local run scripts (later).
- `proto/`: Optional gRPC schemas (later).
- `internal/`: Shared Go packages.

## MVP Interfaces (HTTP+JSON)

### Replay buffer

- `POST /enqueue`
  - Body: `{ "batch_sent_at_ms": int64, "trajectories": [...] }`
- `GET /dequeue?batch_size=...`
  - Response: `{ "trajectories": [...] }`

### Trainer policy

- `GET /policy`
  - Response: `{ "version": int, "updated_at_ms": int64, "weights": { ... } }`

## Trajectory schema (draft)

```
Trajectory {
  worker_id: string
  episode_id: int
  created_at_ms: int64
  episode_reward: float
  steps: [
    {
      obs: [float, float, float, float]
      action: int
      reward: float
      done: bool
      log_prob: float
      value: float
    }
  ]
}
```

## MVP Loop

1. Trainer starts with initial policy and serves `/policy`.
2. Workers pull policy every N seconds, run episodes, batch trajectories.
3. Workers send trajectories to `/enqueue` on the replay buffer.
4. Trainer pulls batches from `/dequeue`, runs PPO update, bumps policy version.
5. Workers pull updated policy and continue.

## Next steps

- Flesh out Go services.
- Implement minimal PPO trainer.
- Add Prometheus metrics + CLI watcher.
