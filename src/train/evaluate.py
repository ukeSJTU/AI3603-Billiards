import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Type

import yaml

from src.train.agents import Agent, BasicAgent, GeometryAgent, RandomAgent
from src.train.poolenv import PoolEnv
from src.utils.logger import get_logger, setup_logger
from src.utils.seed import set_random_seed

logger = get_logger()

WIN_SCORE = 1.0
DRAW_SCORE = 0.5

AGENT_REGISTRY: dict[str, Type[Agent]] = {
    "BasicAgent": BasicAgent,
    "RandomAgent": RandomAgent,
    "GeometryAgent": GeometryAgent,
    # 示例：添加自定义 Agent
    # "MyPPOAgent": MyPPOAgent,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="台球 AI 对战评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 实验配置文件（主要参数）
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/demo.yaml",
        help="实验配置 YAML 文件路径",
    )

    parser.add_argument(
        "--n_games",
        "-n",
        type=int,
        default=None,
        help="评估比赛局数，覆盖配置文件中的 n_games 参数。建议设置为4的倍数以均衡球型分配。",
    )

    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default=None,
        help="实验名称，覆盖配置文件中的 experiment_name 参数。",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="随机种子，覆盖配置文件中的 random_seed 参数。",
    )

    parser.add_argument(
        "--random_seed_enabled",
        action="store_true",
        help="启用随机种子，覆盖配置文件中的 random_seed_enabled 参数。",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """从 YAML 文件加载完整实验配置

    Args:
        config_path: YAML 配置文件路径

    Returns:
        配置字典，包含 agent_a, agent_b, n_games 等所有参数
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not config:
                raise ValueError(f"配置文件为空: {config_path}")
            return config
    except Exception as e:
        raise ValueError(f"解析配置文件失败 {config_path}: {e}")


def merge_config(config: Dict, args: argparse.Namespace) -> Dict:
    """用命令行参数覆盖配置文件（仅覆盖非 None 值）"""
    args_dict = vars(args)

    for key, value in args_dict.items():
        if key != "config" and value is not None:
            # We don't use logger here because logger may not be set up yet
            print(f"覆盖配置: {key} = {value}")
            config[key] = value

    return config


def main() -> Dict[str, int]:
    args = parse_args()

    config = load_config(args.config)

    config = merge_config(config, args)

    folder_name = config.get("experiment_name", "") + f"{datetime.now().strftime('_%Y%m%d_%H%M%S')}"
    n_games = config.get("n_games", 120)

    agent_a_config = config.get("agent_a", {})
    agent_b_config = config.get("agent_b", {})

    # Setup the logger
    setup_logger(
        log_dir=f"experiments/{folder_name}",
        log_filename="evaluation.log",
    )

    logger.info(
        f"Config loaded from yaml file {args.config} and command line args {args}: {config}"
    )

    # We need to save the config used for this experiment
    config_save_path = Path(f"experiments/{folder_name}/config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    set_random_seed(
        enable=config.get("random_seed_enabled", False),
        seed=config.get("random_seed", 42),
    )

    env = PoolEnv()

    results = {
        "AGENT_A_WIN": 0,
        "AGENT_B_WIN": 0,
        "SAME": 0,
        "AGENT_A_SCORE": 0.0,  # 最终得分，胜一局得1分，平局得0.5分
        "AGENT_B_SCORE": 0.0,
    }

    agent_a = AGENT_REGISTRY[agent_a_config.get("type")](**agent_a_config.get("params", {}))
    agent_b = AGENT_REGISTRY[agent_b_config.get("type")](**agent_b_config.get("params", {}))

    players = [agent_a, agent_b]
    target_ball_choice = ["solid", "solid", "stripe", "stripe"]  # 轮换球型

    for i in range(n_games):
        logger.info(f"第 {i} 局比赛开始".center(40, "="))

        target_ball_type = target_ball_choice[i % len(target_ball_choice)]
        logger.info(f"本局比赛 player A 的目标球型: {target_ball_type}")

        env.reset(target_ball=target_ball_type)

        while True:
            player = env.get_curr_player()

            obs = env.get_observation(player)
            if player == "A":
                action = players[i % 2].decision(*obs)
            else:
                action = players[(i + 1) % 2].decision(*obs)
            step_info = env.take_shot(action)

            done, info = env.get_done()

            if done:
                # 统计结果（player A/B 转换为 agent A/B）
                if info["winner"] == "SAME":
                    results["SAME"] += 1
                elif info["winner"] == "A":
                    results[["AGENT_A_WIN", "AGENT_B_WIN"][i % 2]] += 1
                else:
                    results[["AGENT_A_WIN", "AGENT_B_WIN"][(i + 1) % 2]] += 1
                break
            else:
                # TODO: extract more step info if needed
                if step_info.get("ENEMY_INTO_POCKET"):
                    logger.info(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")

    results["AGENT_A_SCORE"] = results["AGENT_A_WIN"] * WIN_SCORE + results["SAME"] * DRAW_SCORE
    results["AGENT_B_SCORE"] = results["AGENT_B_WIN"] * WIN_SCORE + results["SAME"] * DRAW_SCORE

    results_save_path = Path(f"experiments/{folder_name}/results.json")
    with open(results_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    results = main()
    logger.info(f"Evaluation completed. Results: {results}")
