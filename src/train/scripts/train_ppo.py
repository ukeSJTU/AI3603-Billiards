"""
PPO Training Script for Billiards Agent

Self-play training loop with periodic evaluation and checkpointing.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.train.agents.ppo import PPOAgent
from src.train.agents.ppo.ppo_trainer import SelfPlayTrainer
from src.train.poolenv import PoolEnv
from src.utils.logger import get_logger, setup_logger
from src.utils.seed import set_random_seed

setup_logger(
    console_level="WARNING",
)

logger = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO Training for Billiards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c", type=str, default="configs/ppo_training.yaml", help="Training config file"
    )

    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        if not config:
            raise ValueError(f"Empty config file: {config_path}")
        return config


def collect_self_play_episode(
    env: PoolEnv,
    agent_a_policy,
    agent_b_policy,
    agent_a_value,
    agent_b_value,
    reward_scheduler,
    iteration: int,
    device: str,
) -> tuple:
    """
    Collect one self-play episode

    Returns:
        (trajectories_a, trajectories_b, game_result)
    """
    # Reset environment
    target_ball = np.random.choice(["solid", "stripe"])
    env.reset(target_ball=target_ball)

    # Storage for both agents
    traj_a = {"states": [], "actions": [], "log_probs": [], "rewards": [], "values": []}
    traj_b = {"states": [], "actions": [], "log_probs": [], "rewards": [], "values": []}

    done = False
    step_count = 0

    while not done:
        player = env.get_curr_player()  # 'A' or 'B'
        obs = env.get_observation(player)
        balls, my_targets, table = obs

        # Select agent and storage
        if player == "A":
            policy = agent_a_policy
            value_net = agent_a_value
            traj = traj_a
        else:
            policy = agent_b_policy
            value_net = agent_b_value
            traj = traj_b

        # Encode state
        ppo_agent = PPOAgent()  # Temporary for encoding
        state = ppo_agent._encode_state(balls, my_targets, table)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Get action
        with torch.no_grad():
            action_raw, log_prob, _ = policy.sample_action(state_tensor)
            value = value_net(state_tensor).item()

        # Map to game action
        action = ppo_agent._map_action(action_raw.squeeze(0))

        # Execute action
        shot_result = env.take_shot(action)

        # Check if done
        done, game_result = env.get_done()

        # Compute reward
        reward = reward_scheduler.compute_reward(
            shot_result, game_result if done else None, iteration, player
        )

        # Store transition
        traj["states"].append(state_tensor.squeeze(0))
        traj["actions"].append(action_raw.squeeze(0))
        traj["log_probs"].append(log_prob.squeeze(0))
        traj["rewards"].append(reward)
        traj["values"].append(value)

        step_count += 1

        if step_count > 500:  # Safety limit
            logger.warning("Episode exceeded 500 steps, terminating")
            done = True
            game_result = {"winner": "SAME", "hit_count": step_count}

    return traj_a, traj_b, game_result


def evaluate_agent(agent: PPOAgent, opponent_name: str, n_games: int = 20) -> dict:
    """
    Evaluate agent against a baseline opponent

    Returns:
        Dict with win_rate, avg_shots, etc.
    """
    from src.train.agents import BasicAgent, GeometryAgent, RandomAgent

    # Create opponent
    if opponent_name == "RandomAgent":
        opponent = RandomAgent()
    elif opponent_name == "BasicAgent":
        opponent = BasicAgent(initial_search=20, opt_search=10)
    elif opponent_name == "GeometryAgent":
        opponent = GeometryAgent(n_candidates=30, enable_adaptive=True)
    else:
        raise ValueError(f"Unknown opponent: {opponent_name}")

    env = PoolEnv()
    wins = 0
    total_shots = 0

    for i in range(n_games):
        target_ball = "solid" if i % 2 == 0 else "stripe"
        env.reset(target_ball=target_ball)

        done = False
        while not done:
            player = env.get_curr_player()
            obs = env.get_observation(player)

            # Agent A is PPO, Agent B is baseline
            if (i % 2 == 0 and player == "A") or (i % 2 == 1 and player == "B"):
                action = agent.decision(*obs)
            else:
                action = opponent.decision(*obs)

            env.take_shot(action)
            done, info = env.get_done()

        # Check if PPO won
        if (i % 2 == 0 and info["winner"] == "A") or (i % 2 == 1 and info["winner"] == "B"):
            wins += 1

        total_shots += info["hit_count"]

    return {
        "win_rate": wins / n_games,
        "avg_shots": total_shots / n_games,
    }


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment_name", "ppo_training")
    log_dir = f"experiments/{experiment_name}_{timestamp}"
    setup_logger(log_dir=log_dir, log_filename="training.log")

    logger.info(f"Starting PPO training: {experiment_name}")
    logger.info(f"Config: {config}")

    # Set random seed
    set_random_seed(
        enable=config.get("random_seed_enabled", False), seed=config.get("random_seed", 42)
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize trainer
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", "checkpoints/ppo")
    trainer = SelfPlayTrainer(config, checkpoint_dir=checkpoint_dir)

    # TensorBoard
    writer = SummaryWriter(f"runs/{experiment_name}_{timestamp}")

    # Resume if requested
    start_iteration = 0
    if args.resume:
        start_iteration = trainer.load_checkpoint(args.resume)

    # Training loop
    training_config = config.get("training", {})
    n_iterations = training_config.get("n_iterations", 1000)
    games_per_iteration = training_config.get("games_per_iteration", 10)

    eval_config = config.get("evaluation", {})
    eval_interval = eval_config.get("eval_interval", 50)
    eval_games = eval_config.get("eval_games", 20)
    eval_opponents = eval_config.get("opponents", ["RandomAgent", "BasicAgent"])

    checkpoint_interval = config.get("checkpointing", {}).get("checkpoint_interval", 100)
    snapshot_interval = config.get("self_play", {}).get("snapshot_interval", 50)

    env = PoolEnv()

    for iteration in range(start_iteration, n_iterations):
        logger.info(f"=== Iteration {iteration}/{n_iterations} ===")

        # Clear buffer
        trainer.buffer.clear()

        # Load opponent (or use current policy early on)
        opponent_policy = trainer.load_random_opponent()
        if opponent_policy is None:
            opponent_policy = trainer.policy_net
            logger.info("Early training: using current policy as opponent")

        # Collect self-play games
        game_results = []
        for game_idx in range(games_per_iteration):
            traj_a, traj_b, game_result = collect_self_play_episode(
                env,
                trainer.policy_net,
                opponent_policy,
                trainer.value_net,
                trainer.value_net,
                trainer.reward_scheduler,
                iteration,
                device,
            )

            # Add trajectories to buffer (only from current policy, not opponent)
            for i in range(len(traj_a["states"])):
                trainer.buffer.add(
                    traj_a["states"][i],
                    traj_a["actions"][i],
                    traj_a["log_probs"][i],
                    traj_a["rewards"][i],
                    traj_a["values"][i],
                    done=(i == len(traj_a["states"]) - 1),
                )

            game_results.append(game_result)

        # Perform PPO update
        if len(trainer.buffer) > 0:
            metrics = trainer.update_ppo()

            # Log training metrics
            writer.add_scalar("Loss/policy", metrics["policy_loss"], iteration)
            writer.add_scalar("Loss/value", metrics["value_loss"], iteration)
            writer.add_scalar("Loss/entropy", metrics["entropy"], iteration)
            writer.add_scalar("PPO/approx_kl", metrics["approx_kl"], iteration)
            writer.add_scalar("PPO/clipfrac", metrics["clipfrac"], iteration)

            # Log reward statistics
            all_rewards = trainer.buffer.get()["rewards"]
            writer.add_scalar("Reward/mean", np.mean(all_rewards), iteration)
            writer.add_scalar("Reward/std", np.std(all_rewards), iteration)

            # Log dense weight
            dense_weight = trainer.reward_scheduler.get_dense_weight(iteration)
            writer.add_scalar("Reward/dense_weight", dense_weight, iteration)

            logger.info(
                f"PPO Update: policy_loss={metrics['policy_loss']:.4f}, "
                f"value_loss={metrics['value_loss']:.4f}, "
                f"entropy={metrics['entropy']:.4f}, "
                f"kl={metrics['approx_kl']:.4f}"
            )

        # Evaluation
        if iteration % eval_interval == 0 and iteration > 0:
            logger.info(f"Evaluating at iteration {iteration}")
            eval_agent = PPOAgent(deterministic=True, device=device)
            eval_agent.policy_net = trainer.policy_net

            for opponent_name in eval_opponents:
                eval_metrics = evaluate_agent(eval_agent, opponent_name, n_games=eval_games)
                writer.add_scalar(
                    f"Eval/win_rate_vs_{opponent_name.lower()}",
                    eval_metrics["win_rate"],
                    iteration,
                )
                writer.add_scalar(
                    f"Eval/avg_shots_vs_{opponent_name.lower()}",
                    eval_metrics["avg_shots"],
                    iteration,
                )
                logger.info(
                    f"  vs {opponent_name}: {eval_metrics['win_rate']:.1%} "
                    f"({eval_metrics['avg_shots']:.1f} shots/game)"
                )

        # Update opponent pool
        if iteration % snapshot_interval == 0 and iteration > 0:
            trainer.save_opponent_snapshot(iteration)

        # Checkpointing
        if iteration % checkpoint_interval == 0 and iteration > 0:
            trainer.save_checkpoint(iteration, metrics if len(trainer.buffer) > 0 else {})

    # Save final model
    final_path = Path(checkpoint_dir) / "best_model.pth"
    trainer.save_checkpoint(n_iterations, {})
    logger.info(f"Training complete! Final model saved to {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
