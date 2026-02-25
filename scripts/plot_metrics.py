import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot recorded training metrics over time.")
    parser.add_argument("--input", default="metrics.csv")
    parser.add_argument("--output", default="metrics.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        raise SystemExit("matplotlib is required: pip install matplotlib")

    elapsed = []
    reward = []
    rolling = []

    with open(args.input, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            elapsed.append(float(row.get("elapsed_sec", 0.0)))
            reward.append(float(row.get("last_batch_reward_mean", 0.0) or 0.0))
            rolling.append(float(row.get("rolling_reward_mean", 0.0) or 0.0))

    plt.figure(figsize=(9, 4))
    plt.plot(elapsed, reward, label="batch reward", alpha=0.4)
    plt.plot(elapsed, rolling, label="rolling reward", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.title("CartPole Training Reward vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
