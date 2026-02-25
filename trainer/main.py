import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BUFFER_URL = "http://localhost:9001"
DEFAULT_PORT = 9002
DEFAULT_BATCH_SIZE = 128
DEFAULT_POLL_MS = 200
DEFAULT_GAMMA = 0.99
DEFAULT_CLIP_EPS = 0.2
DEFAULT_POLICY_LR = 0.002
DEFAULT_VALUE_LR = 0.01
DEFAULT_EPOCHS = 4
DEFAULT_MINIBATCH_SIZE = 128
DEFAULT_ENTROPY_COEF = 0.01


@dataclass
class TrainerConfig:
    batch_size: int
    poll_ms: int
    gamma: float
    clip_eps: float
    policy_lr: float
    value_lr: float
    epochs: int
    minibatch_size: int
    entropy_coef: float
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "batch_size": self.batch_size,
                "poll_ms": self.poll_ms,
                "gamma": self.gamma,
                "clip_eps": self.clip_eps,
                "policy_lr": self.policy_lr,
                "value_lr": self.value_lr,
                "epochs": self.epochs,
                "minibatch_size": self.minibatch_size,
                "entropy_coef": self.entropy_coef,
            }

    def update(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            if "batch_size" in payload:
                self.batch_size = int(payload["batch_size"])
            if "poll_ms" in payload:
                self.poll_ms = int(payload["poll_ms"])
            if "gamma" in payload:
                self.gamma = float(payload["gamma"])
            if "clip_eps" in payload:
                self.clip_eps = float(payload["clip_eps"])
            if "policy_lr" in payload:
                self.policy_lr = float(payload["policy_lr"])
            if "value_lr" in payload:
                self.value_lr = float(payload["value_lr"])
            if "epochs" in payload:
                self.epochs = int(payload["epochs"])
            if "minibatch_size" in payload:
                self.minibatch_size = int(payload["minibatch_size"])
            if "entropy_coef" in payload:
                self.entropy_coef = float(payload["entropy_coef"])


@dataclass
class PolicyState:
    weights: Dict[str, Any]
    version: int = 0
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    last_batch_trajectories: int = 0
    last_batch_steps: int = 0
    last_batch_reward_mean: float = 0.0
    last_batch_at_ms: int = 0
    rolling_reward_mean: float = 0.0
    rolling_reward_alpha: float = 0.1
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "version": self.version,
                "updated_at_ms": self.updated_at_ms,
                "weights": self.weights,
            }

    def mark_update(self) -> None:
        with self.lock:
            self.version += 1
            self.updated_at_ms = int(time.time() * 1000)

    def stats_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "last_batch_trajectories": self.last_batch_trajectories,
                "last_batch_steps": self.last_batch_steps,
                "last_batch_reward_mean": self.last_batch_reward_mean,
                "last_batch_at_ms": self.last_batch_at_ms,
                "rolling_reward_mean": self.rolling_reward_mean,
            }

    def update_weights(self, weights: Dict[str, Any]) -> None:
        with self.lock:
            self.weights = weights
            self.version += 1
            self.updated_at_ms = int(time.time() * 1000)

    def update_stats(self, trajectories: int, steps: int, reward_mean: float) -> None:
        with self.lock:
            self.last_batch_trajectories = trajectories
            self.last_batch_steps = steps
            self.last_batch_reward_mean = reward_mean
            self.last_batch_at_ms = int(time.time() * 1000)
            if self.rolling_reward_mean == 0.0:
                self.rolling_reward_mean = reward_mean
            else:
                alpha = self.rolling_reward_alpha
                self.rolling_reward_mean = (1 - alpha) * self.rolling_reward_mean + alpha * reward_mean


def default_weights() -> Dict[str, Any]:
    return {
        "w": [[0.01, 0.01, 0.01, 0.01], [-0.01, -0.01, -0.01, -0.01]],
        "b": [0.0, 0.0],
        "vw": [0.0, 0.0, 0.0, 0.0],
        "vb": 0.0,
    }


def _softmax(logits: List[float]) -> List[float]:
    max_logit = max(logits)
    exp_values = [math.exp(val - max_logit) for val in logits]
    total = sum(exp_values)
    return [val / total for val in exp_values]


def _policy_logits(weights: Dict[str, Any], obs: List[float]) -> List[float]:
    logits = [weights["b"][0], weights["b"][1]]
    for idx in range(2):
        logits[idx] += sum(w * o for w, o in zip(weights["w"][idx], obs))
    return logits


def _log_prob(weights: Dict[str, Any], obs: List[float], action: int) -> Tuple[float, List[float]]:
    logits = _policy_logits(weights, obs)
    probs = _softmax(logits)
    return math.log(probs[action] + 1e-8), probs


def _entropy(probs: List[float]) -> float:
    return -sum(p * math.log(p + 1e-8) for p in probs)


def _compute_returns(steps: List[Dict[str, Any]], gamma: float) -> List[float]:
    returns = [0.0] * len(steps)
    running_return = 0.0
    for idx in range(len(steps) - 1, -1, -1):
        step = steps[idx]
        if step.get("done"):
            running_return = 0.0
        running_return = step.get("reward", 0.0) + gamma * running_return
        returns[idx] = running_return
    return returns


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    mean = sum(values) / len(values)
    variance = sum((val - mean) ** 2 for val in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(val - mean) / std for val in values]


def ppo_update(
    trajectories: List[Dict[str, Any]],
    weights: Dict[str, Any],
    gamma: float,
    clip_eps: float,
    policy_lr: float,
    value_lr: float,
    epochs: int,
    minibatch_size: int,
    entropy_coef: float,
) -> Dict[str, Any]:
    samples: List[Tuple[List[float], int, float, float]] = []
    advantages: List[float] = []

    for traj in trajectories:
        steps = traj.get("steps", [])
        if not steps:
            continue
        returns = _compute_returns(steps, gamma)
        for step, ret in zip(steps, returns):
            obs = step.get("obs", [])
            if len(obs) != 4:
                continue
            action = int(step.get("action", 0))
            old_log_prob = float(step.get("log_prob", 0.0))
            value = float(step.get("value", 0.0))
            advantage = ret - value
            samples.append((obs, action, old_log_prob, ret))
            advantages.append(advantage)

    if not samples:
        return weights

    normalized_advantages = _normalize(advantages)
    epochs = max(1, epochs)
    minibatch_size = max(1, min(minibatch_size, len(samples)))

    indices = list(range(len(samples)))
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(samples), minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            grad_w = [[0.0 for _ in range(4)] for _ in range(2)]
            grad_b = [0.0, 0.0]
            grad_vw = [0.0, 0.0, 0.0, 0.0]
            grad_vb = 0.0
            entropy_sum = 0.0

            for idx in batch_indices:
                obs, action, old_log_prob, ret = samples[idx]
                advantage = normalized_advantages[idx]
                new_log_prob, probs = _log_prob(weights, obs, action)
                ratio = math.exp(new_log_prob - old_log_prob)
                if advantage >= 0:
                    clipped_ratio = min(ratio, 1.0 + clip_eps)
                else:
                    clipped_ratio = max(ratio, 1.0 - clip_eps)
                weight = clipped_ratio * advantage

                for jdx in range(2):
                    indicator = 1.0 if jdx == action else 0.0
                    coeff = (indicator - probs[jdx]) * weight
                    grad_b[jdx] += coeff
                    for kdx, obs_val in enumerate(obs):
                        grad_w[jdx][kdx] += coeff * obs_val

                value_est = weights["vb"] + sum(w * o for w, o in zip(weights["vw"], obs))
                value_error = ret - value_est
                grad_vb += value_error
                for kdx, obs_val in enumerate(obs):
                    grad_vw[kdx] += value_error * obs_val

                entropy_sum += _entropy(probs)

            batch_count = float(len(batch_indices))
            entropy_grad = entropy_coef * (entropy_sum / batch_count)
            for jdx in range(2):
                weights["b"][jdx] += policy_lr * (grad_b[jdx] / batch_count)
                for kdx in range(4):
                    weights["w"][jdx][kdx] += policy_lr * (grad_w[jdx][kdx] / batch_count)
                weights["b"][jdx] += policy_lr * entropy_grad
            for kdx in range(4):
                weights["vw"][kdx] += value_lr * (grad_vw[kdx] / batch_count)
            weights["vb"] += value_lr * (grad_vb / batch_count)

    return weights


class PolicyHandler(BaseHTTPRequestHandler):
    policy_state: PolicyState = None
    trainer_config: TrainerConfig = None

    def do_GET(self) -> None:
        if self.path.startswith("/policy"):
            payload = self.policy_state.snapshot()
            self._write_json(200, payload)
            return
        if self.path.startswith("/stats"):
            payload = self.policy_state.stats_snapshot()
            self._write_json(200, payload)
            return
        if self.path.startswith("/config"):
            if self.command != "GET":
                self.send_response(405)
                self.end_headers()
                return
            payload = self.trainer_config.snapshot()
            self._write_json(200, payload)
            return
        if self.path.startswith("/healthz"):
            self._write_text(200, "ok")
            return
        self.send_response(404)
        self.end_headers()

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, status: int, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), format % args)

    def do_POST(self) -> None:
        if self.path.startswith("/config"):
            payload = self._read_json()
            if payload is None:
                return
            try:
                self.trainer_config.update(payload)
            except (TypeError, ValueError):
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def _read_json(self) -> Dict[str, Any] | None:
        length = self.headers.get("Content-Length")
        if not length:
            self.send_response(400)
            self.end_headers()
            return None
        try:
            data = self.rfile.read(int(length)).decode("utf-8")
            return json.loads(data)
        except (ValueError, json.JSONDecodeError):
            self.send_response(400)
            self.end_headers()
            return None


def dequeue_batch(buffer_url: str, batch_size: int) -> List[Dict[str, Any]]:
    url = f"{buffer_url}/dequeue?batch_size={batch_size}"
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=5) as response:
            if response.status == 204:
                return []
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("trajectories", [])
    except HTTPError as exc:
        if exc.code == 204:
            return []
        logging.warning("dequeue failed: %s", exc)
        return []
    except URLError as exc:
        logging.warning("buffer unreachable: %s", exc)
        return []


def training_loop(policy: PolicyState, config: TrainerConfig, buffer_url: str) -> None:
    while True:
        config_snapshot = config.snapshot()
        trajectories = dequeue_batch(buffer_url, int(config_snapshot["batch_size"]))
        if trajectories:
            step_count = sum(len(traj.get("steps", [])) for traj in trajectories)
            rewards = [float(traj.get("episode_reward", 0.0)) for traj in trajectories]
            reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
            logging.info("trainer batch: trajectories=%d steps=%d", len(trajectories), step_count)
            current_weights = policy.snapshot()["weights"]
            updated_weights = ppo_update(
                trajectories,
                current_weights,
                float(config_snapshot["gamma"]),
                float(config_snapshot["clip_eps"]),
                float(config_snapshot["policy_lr"]),
                float(config_snapshot["value_lr"]),
                int(config_snapshot["epochs"]),
                int(config_snapshot["minibatch_size"]),
                float(config_snapshot["entropy_coef"]),
            )
            policy.update_weights(updated_weights)
            policy.update_stats(len(trajectories), step_count, reward_mean)
        time.sleep(int(config_snapshot["poll_ms"]) / 1000.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[trainer] %(message)s")

    buffer_url = os.getenv("BUFFER_URL", DEFAULT_BUFFER_URL)
    port = int(os.getenv("PORT", DEFAULT_PORT))
    config = TrainerConfig(
        batch_size=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        poll_ms=int(os.getenv("POLL_INTERVAL_MS", DEFAULT_POLL_MS)),
        gamma=float(os.getenv("GAMMA", DEFAULT_GAMMA)),
        clip_eps=float(os.getenv("CLIP_EPS", DEFAULT_CLIP_EPS)),
        policy_lr=float(os.getenv("POLICY_LR", DEFAULT_POLICY_LR)),
        value_lr=float(os.getenv("VALUE_LR", DEFAULT_VALUE_LR)),
        epochs=int(os.getenv("PPO_EPOCHS", DEFAULT_EPOCHS)),
        minibatch_size=int(os.getenv("PPO_MINIBATCH", DEFAULT_MINIBATCH_SIZE)),
        entropy_coef=float(os.getenv("ENTROPY_COEF", DEFAULT_ENTROPY_COEF)),
    )

    policy_state = PolicyState(weights=default_weights())

    PolicyHandler.policy_state = policy_state
    PolicyHandler.trainer_config = config
    server = ThreadingHTTPServer(("", port), PolicyHandler)

    thread = threading.Thread(
        target=training_loop,
        args=(policy_state, config, buffer_url),
        daemon=True,
    )
    thread.start()

    logging.info("trainer listening on :%d", port)
    logging.info("buffer URL: %s", buffer_url)
    server.serve_forever()


if __name__ == "__main__":
    main()
