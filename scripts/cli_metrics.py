import argparse
import json
import os
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BUFFER_URL = "http://localhost:9001"
DEFAULT_TRAINER_URL = "http://localhost:9002"


def fetch_json(url: str) -> Optional[Dict[str, Any]]:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=2) as response:
            if response.status == 204:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError):
        return None


def format_age(ms: Optional[int]) -> str:
    if ms is None or ms < 0:
        return "-"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def render_line(
    buffer_stats: Optional[Dict[str, Any]],
    trainer_policy: Optional[Dict[str, Any]],
    trainer_stats: Optional[Dict[str, Any]],
    trainer_config: Optional[Dict[str, Any]],
    buffer_config: Optional[Dict[str, Any]],
) -> str:
    now_ms = int(time.time() * 1000)

    queue_length = buffer_stats.get("queue_length") if buffer_stats else "-"
    capacity = buffer_stats.get("capacity") if buffer_stats else "-"
    policy = buffer_stats.get("policy") if buffer_stats else "-"

    version = trainer_policy.get("version") if trainer_policy else "-"
    updated_at_ms = trainer_policy.get("updated_at_ms") if trainer_policy else None
    policy_age = format_age(now_ms - updated_at_ms) if isinstance(updated_at_ms, int) else "-"

    last_traj = trainer_stats.get("last_batch_trajectories") if trainer_stats else 0
    last_steps = trainer_stats.get("last_batch_steps") if trainer_stats else 0
    last_reward = trainer_stats.get("last_batch_reward_mean") if trainer_stats else None
    last_reward_str = f"{last_reward:.2f}" if isinstance(last_reward, (int, float)) else "-"
    rolling_reward = trainer_stats.get("rolling_reward_mean") if trainer_stats else None
    rolling_reward_str = f"{rolling_reward:.2f}" if isinstance(rolling_reward, (int, float)) else "-"
    avg_steps = "-"
    if isinstance(last_traj, int) and last_traj > 0 and isinstance(last_steps, int):
        avg_steps = f"{last_steps / last_traj:.1f}"
    last_at_ms = trainer_stats.get("last_batch_at_ms") if trainer_stats else None
    last_age = format_age(now_ms - last_at_ms) if isinstance(last_at_ms, int) and last_at_ms > 0 else "-"

    batch_size = trainer_config.get("batch_size") if trainer_config else "-"
    poll_ms = trainer_config.get("poll_ms") if trainer_config else "-"
    gamma = trainer_config.get("gamma") if trainer_config else "-"
    clip_eps = trainer_config.get("clip_eps") if trainer_config else "-"
    policy_lr = trainer_config.get("policy_lr") if trainer_config else "-"
    value_lr = trainer_config.get("value_lr") if trainer_config else "-"
    config_line = (
        f"cfg batch={batch_size} poll={poll_ms}ms gamma={gamma} clip={clip_eps} "
        f"lr={policy_lr}/{value_lr}"
    )
    buffer_policy = buffer_config.get("policy") if buffer_config else policy
    buffer_capacity = buffer_config.get("capacity") if buffer_config else capacity
    buffer_line = f"buffer {queue_length}/{buffer_capacity} ({buffer_policy})"

    timestamp = time.strftime("%H:%M:%S")
    return (
        f"{timestamp} | {buffer_line}"
        f" | policy v{version} age {policy_age}"
        f" | batch traj={last_traj} steps={last_steps} avg_steps={avg_steps} "
        f"reward={last_reward_str} roll={rolling_reward_str} age={last_age}"
        f" | {config_line}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI metrics viewer for the MVP trainer and replay buffer.")
    parser.add_argument("--buffer-url", default=os.getenv("BUFFER_URL", DEFAULT_BUFFER_URL))
    parser.add_argument("--trainer-url", default=os.getenv("TRAINER_URL", DEFAULT_TRAINER_URL))
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds.")
    args = parser.parse_args()

    print("time | buffer queue/capacity (policy) | policy version age | batch stats | config")
    while True:
        buffer_stats = fetch_json(f"{args.buffer_url}/stats")
        buffer_config = fetch_json(f"{args.buffer_url}/config")
        trainer_policy = fetch_json(f"{args.trainer_url}/policy")
        trainer_stats = fetch_json(f"{args.trainer_url}/stats")
        trainer_config = fetch_json(f"{args.trainer_url}/config")
        print(render_line(buffer_stats, trainer_policy, trainer_stats, trainer_config, buffer_config))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
