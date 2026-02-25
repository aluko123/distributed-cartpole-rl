import argparse
import csv
import json
import os
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Record trainer stats over time to a CSV file.")
    parser.add_argument("--trainer-url", default=os.getenv("TRAINER_URL", DEFAULT_TRAINER_URL))
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds.")
    parser.add_argument("--output", default="metrics.csv")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds; 0 means run until Ctrl+C.")
    args = parser.parse_args()

    start = time.time()
    fieldnames = [
        "elapsed_sec",
        "policy_version",
        "last_batch_reward_mean",
        "rolling_reward_mean",
        "last_batch_steps",
        "last_batch_trajectories",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            elapsed = time.time() - start
            policy = fetch_json(f"{args.trainer_url}/policy") or {}
            stats = fetch_json(f"{args.trainer_url}/stats") or {}

            row = {
                "elapsed_sec": f"{elapsed:.2f}",
                "policy_version": policy.get("version", ""),
                "last_batch_reward_mean": stats.get("last_batch_reward_mean", ""),
                "rolling_reward_mean": stats.get("rolling_reward_mean", ""),
                "last_batch_steps": stats.get("last_batch_steps", ""),
                "last_batch_trajectories": stats.get("last_batch_trajectories", ""),
            }
            writer.writerow(row)
            handle.flush()

            if args.duration > 0 and elapsed >= args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
