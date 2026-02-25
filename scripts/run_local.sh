#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

BUFFER_PORT="${BUFFER_PORT:-9001}"
TRAINER_PORT="${TRAINER_PORT:-9002}"
BUFFER_URL="${BUFFER_URL:-http://localhost:${BUFFER_PORT}}"
TRAINER_URL="${TRAINER_URL:-http://localhost:${TRAINER_PORT}}"
WORKER_COUNT="${WORKER_COUNT:-1}"
RECORD_METRICS="${RECORD_METRICS:-1}"
METRICS_OUTPUT="${METRICS_OUTPUT:-metrics.csv}"
RUN_CLI_METRICS="${RUN_CLI_METRICS:-1}"

cd "${ROOT_DIR}"

echo "Starting replay buffer on ${BUFFER_PORT}"
go run ./cmd/replay-buffer &
BUFFER_PID=$!

echo "Starting trainer on ${TRAINER_PORT}"
BUFFER_URL="${BUFFER_URL}" PORT="${TRAINER_PORT}" python3 trainer/main.py &
TRAINER_PID=$!

echo "Starting ${WORKER_COUNT} worker(s)"
WORKER_PIDS=()
for idx in $(seq 1 "${WORKER_COUNT}"); do
  WORKER_ID="worker-${idx}" \
  BUFFER_URL="${BUFFER_URL}" \
  TRAINER_URL="${TRAINER_URL}" \
  go run ./cmd/rollout-worker &
  WORKER_PIDS+=("$!")
done

METRICS_PID=""
if [[ "${RECORD_METRICS}" == "1" ]]; then
  echo "Recording trainer metrics to ${METRICS_OUTPUT}"
  TRAINER_URL="${TRAINER_URL}" python3 scripts/record_metrics.py --output "${METRICS_OUTPUT}" &
  METRICS_PID=$!
fi

CLI_PID=""
if [[ "${RUN_CLI_METRICS}" == "1" ]]; then
  echo "Starting CLI metrics viewer"
  BUFFER_URL="${BUFFER_URL}" TRAINER_URL="${TRAINER_URL}" python3 scripts/cli_metrics.py &
  CLI_PID=$!
fi

echo "All services started. Press Ctrl+C to stop."

cleanup() {
  echo "Stopping services..."
  if [[ -n "${METRICS_PID}" ]]; then
    kill "${METRICS_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CLI_PID}" ]]; then
    kill "${CLI_PID}" 2>/dev/null || true
  fi
  kill "${WORKER_PIDS[@]}" 2>/dev/null || true
  kill "${TRAINER_PID}" 2>/dev/null || true
  kill "${BUFFER_PID}" 2>/dev/null || true
}

trap cleanup EXIT
wait
