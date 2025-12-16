#!/usr/bin/env python3
"""
evaluate.py - Agent 评估脚本

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配
- 支持命令行参数配置

使用方式：
  python evaluate.py                          # 默认配置
  python evaluate.py -n 10                    # 快速测试（10局）
  python evaluate.py --seed 42 --seed-enabled # 固定种子
  python evaluate.py --agent-b MyAgent -n 60  # 测试自定义 Agent
  python evaluate.py -q                       # 安静模式
"""

import argparse
from typing import Type

# 初始化 logger（在其他导入之前）
try:
    from logger import setup_logger

    setup_logger()
except ImportError:
    pass

from agent import BaseAgent, BasicAgent, NewAgent
from poolenv import PoolEnv
from utils import set_random_seed

# 配置导入
try:
    from config import DEFAULT_EVAL_CONFIG
except ImportError:

    class DEFAULT_EVAL_CONFIG:  # type: ignore
        n_games = 120
        random_seed_enabled = False
        random_seed = 42


# ============ Agent 注册表 ============
# 学生可以在这里注册自定义 Agent
AGENT_REGISTRY: dict[str, Type[BaseAgent]] = {
    "BasicAgent": BasicAgent,
    "NewAgent": NewAgent,
    # 示例：添加自定义 Agent
    # "MyPPOAgent": MyPPOAgent,
    # "MySACAgent": MySACAgent,
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="台球 AI 对战评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 对战配置
    parser.add_argument(
        "--n-games",
        "-n",
        type=int,
        default=DEFAULT_EVAL_CONFIG.n_games,
        help="对战局数",
    )

    # Agent 选择
    parser.add_argument(
        "--agent-a",
        type=str,
        default="BasicAgent",
        choices=list(AGENT_REGISTRY.keys()),
        help="Agent A 类型",
    )
    parser.add_argument(
        "--agent-b",
        type=str,
        default="NewAgent",
        choices=list(AGENT_REGISTRY.keys()),
        help="Agent B 类型",
    )

    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_CONFIG.random_seed,
        help="随机种子值",
    )
    parser.add_argument(
        "--seed-enabled",
        action="store_true",
        default=DEFAULT_EVAL_CONFIG.random_seed_enabled,
        help="启用固定随机种子",
    )

    # 输出控制
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="减少输出信息",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="增加输出信息",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_random_seed(enable=args.seed_enabled, seed=args.seed)

    # 创建 Agent
    agent_a = AGENT_REGISTRY[args.agent_a]()
    agent_b = AGENT_REGISTRY[args.agent_b]()

    # 创建环境
    env = PoolEnv()

    # 结果统计
    results = {"AGENT_A_WIN": 0, "AGENT_B_WIN": 0, "SAME": 0}

    # 对战设置
    players = [agent_a, agent_b]
    target_ball_choice = ["solid", "solid", "stripe", "stripe"]

    # 主循环
    for i in range(args.n_games):
        if not args.quiet:
            print()
            print(f"------- 第 {i} 局比赛开始 -------")

        env.reset(target_ball=target_ball_choice[i % 4])

        if not args.quiet:
            player_class = players[i % 2].__class__.__name__
            ball_type = target_ball_choice[i % 4]
            print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")

        while True:
            player = env.get_curr_player()

            if args.verbose:
                print(f"[第{env.hit_count}次击球] player: {player}")

            obs = env.get_observation(player)

            if player == "A":
                action = players[i % 2].decision(*obs)
            else:
                action = players[(i + 1) % 2].decision(*obs)

            step_info = env.take_shot(action)
            done, info = env.get_done()

            if not done:
                if args.verbose and step_info.get("ENEMY_INTO_POCKET"):
                    print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")

            if done:
                # 统计结果（player A/B 转换为 agent A/B）
                if info["winner"] == "SAME":
                    results["SAME"] += 1
                elif info["winner"] == "A":
                    results[["AGENT_A_WIN", "AGENT_B_WIN"][i % 2]] += 1
                else:
                    results[["AGENT_A_WIN", "AGENT_B_WIN"][(i + 1) % 2]] += 1
                break

    # 计算分数
    results["AGENT_A_SCORE"] = results["AGENT_A_WIN"] * 1 + results["SAME"] * 0.5
    results["AGENT_B_SCORE"] = results["AGENT_B_WIN"] * 1 + results["SAME"] * 0.5

    # 输出结果
    print("\n" + "=" * 50)
    print(f"对战结果 ({args.n_games} 局):")
    print(
        f"  Agent A ({args.agent_a}): {results['AGENT_A_WIN']} 胜, {results['AGENT_A_SCORE']:.1f} 分"
    )
    print(
        f"  Agent B ({args.agent_b}): {results['AGENT_B_WIN']} 胜, {results['AGENT_B_SCORE']:.1f} 分"
    )
    print(f"  平局: {results['SAME']}")
    print("=" * 50)

    # 计算胜率
    win_rate_a = results["AGENT_A_SCORE"] / args.n_games * 100
    win_rate_b = results["AGENT_B_SCORE"] / args.n_games * 100
    print(f"  Agent A 胜率: {win_rate_a:.1f}%")
    print(f"  Agent B 胜率: {win_rate_b:.1f}%")

    return results


if __name__ == "__main__":
    results = main()
