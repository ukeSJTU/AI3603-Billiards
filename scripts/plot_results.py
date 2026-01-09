"""
Quick plotting utilities for GeometryAgent experiments.

Usage:
    python scripts/plot_results.py \
        --experiments experiments \
        --output_dir assets

Outputs:
    - assets/win_rates.png: bar chart of win rate per run (sorted).
    - assets/fouls_vs_own.png: scatter plot of average fouls vs. own_pocket for runs with metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_results(exp_root: Path) -> List[Dict]:
    """Scan experiments folder and load results/metrics if available."""
    runs: List[Dict] = []
    for results_path in exp_root.glob("*/results.json"):
        run_dir = results_path.parent
        cfg_path = run_dir / "config.yaml"
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        total = r.get("AGENT_A_WIN", 0) + r.get("AGENT_B_WIN", 0) + r.get("SAME", 0)
        win_rate = r.get("AGENT_A_WIN", 0) / total if total else 0.0
        seed: Optional[str] = None
        if cfg_path.exists():
            # config.yaml was saved by evaluate.py using yaml.safe_dump
            # Avoid importing yaml; parse the seed line manually.
            for line in cfg_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("random_seed"):
                    seed = line.split(":")[1].strip()
                    break
        run_info = {
            "name": run_dir.name,
            "win_rate": win_rate,
            "A_win": r.get("AGENT_A_WIN", 0),
            "B_win": r.get("AGENT_B_WIN", 0),
            "SAME": r.get("SAME", 0),
            "seed": seed,
        }
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                if metrics:
                    n = len(metrics)
                    own = sum(m.get("own_pocket", 0) for m in metrics) / n
                    enemy = sum(m.get("enemy_pocket", 0) for m in metrics) / n
                    fouls = (
                        sum(
                            m.get("white_in", 0)
                            + m.get("foul_first_hit", 0)
                            + m.get("no_pocket_no_rail", 0)
                            + m.get("no_hit", 0)
                            for m in metrics
                        )
                        / n
                    )
                    run_info.update({"own_pocket": own, "enemy_pocket": enemy, "fouls": fouls})
            except Exception:
                pass
        runs.append(run_info)
    return sorted(runs, key=lambda x: x["win_rate"], reverse=True)


def plot_win_rates(runs: List[Dict], out_path: Path) -> None:
    names = [r["name"] for r in runs]
    win_rates = [r["win_rate"] for r in runs]
    seeds = [r.get("seed") or "" for r in runs]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(range(len(names)), win_rates, color="#4e79a7")
    plt.xticks(range(len(names)), names, rotation=75, ha="right", fontsize=8)
    plt.ylabel("Win Rate (Agent A)")
    plt.ylim(0, 1.0)
    plt.title("Evaluation Win Rates (sorted)")
    for idx, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{win_rates[idx]:.2f}\\nseed={seeds[idx]}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_fouls_vs_own(runs: List[Dict], out_path: Path) -> None:
    runs_with_metrics = [r for r in runs if "own_pocket" in r and "fouls" in r]
    if not runs_with_metrics:
        return
    plt.figure(figsize=(6, 5))
    for r in runs_with_metrics:
        plt.scatter(r["fouls"], r["own_pocket"], s=50, label=r["name"])
        plt.text(r["fouls"] + 0.02, r["own_pocket"], r.get("seed") or "", fontsize=7)
    plt.xlabel("Avg Fouls per Game (white + first-hit + no-rail + no-hit)")
    plt.ylabel("Avg Own Pockets per Game")
    plt.title("Fouls vs. Own Pockets")
    plt.legend(fontsize=7, frameon=False, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results from experiments/*")
    parser.add_argument(
        "--experiments", type=str, default="experiments", help="Path to experiments root"
    )
    parser.add_argument(
        "--output_dir", type=str, default="assets", help="Directory to save figures"
    )
    args = parser.parse_args()

    exp_root = Path(args.experiments)
    out_dir = Path(args.output_dir)
    runs = load_results(exp_root)
    if not runs:
        print("No experiments found.")
        return

    plot_win_rates(runs, out_dir / "win_rates.png")
    plot_fouls_vs_own(runs, out_dir / "fouls_vs_own.png")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
