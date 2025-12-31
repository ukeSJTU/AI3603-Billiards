"""
replay_viewer.py - 回放可视化工具

用于加载和可视化评估过程中保存的回放记录。

使用方法：
    python replay_viewer.py --replay experiments/demo_20251231/replay.msgpack
    python replay_viewer.py -r experiments/demo_20251231/replay.msgpack -n 5  # 从第5轮开始
"""

import argparse
import sys
from pathlib import Path

import pooltool as pt


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="台球对局回放可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--replay",
        "-r",
        type=str,
        required=True,
        help="回放文件路径（replay.msgpack 或 replay.json）",
    )

    parser.add_argument(
        "--start_round",
        "-n",
        type=int,
        default=0,
        help="开始回放的轮次（0表示从第一轮开始）",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查文件是否存在
    replay_path = Path(args.replay)
    if not replay_path.exists():
        print(f"错误：回放文件不存在: {args.replay}")
        sys.exit(1)

    # 加载回放记录
    print(f"正在加载回放文件: {args.replay}")
    try:
        multisystem = pt.MultiSystem.load(str(replay_path))
    except Exception as e:
        print(f"错误：无法加载回放文件: {e}")
        sys.exit(1)

    total_shots = len(multisystem)
    print(f"回放文件加载成功，共 {total_shots} 个击球记录")

    # 验证起始轮次
    if args.start_round < 0:
        print("警告：起始轮次不能为负数，将从第 0 轮开始")
        start_round = 0
    elif args.start_round >= total_shots:
        print(
            f"警告：起始轮次 {args.start_round} 超出范围（共 {total_shots} 轮），将从第 {total_shots - 1} 轮开始"
        )
        start_round = total_shots - 1
    else:
        start_round = args.start_round

    # 创建从指定轮次开始的子集
    if start_round > 0:
        print(f"从第 {start_round} 轮开始回放（跳过前 {start_round} 轮）")
        # 创建新的 MultiSystem，只包含从 start_round 开始的记录
        filtered_multisystem = pt.MultiSystem()
        for i in range(start_round, total_shots):
            filtered_multisystem.append(multisystem[i])
        multisystem = filtered_multisystem
        print(f"当前回放包含 {len(multisystem)} 个击球记录")

    # 启动可视化
    print("\n启动可视化界面...")
    print("操作提示：")
    print("  - 按 'n' 键：下一个击球")
    print("  - 按 'p' 键：上一个击球")
    print("  - 按 Enter：切换并行可视化模式")
    print("  - 按 ESC：退出\n")

    try:
        pt.show(
            multisystem,
            title=f"回放查看器 - 共 {len(multisystem)} 个击球记录（按 'n'/'p' 切换，ESC 退出）",
        )
    except KeyboardInterrupt:
        print("\n用户中断，退出回放")
    except Exception as e:
        print(f"\n可视化过程中发生错误: {e}")
        sys.exit(1)

    print("回放结束")


if __name__ == "__main__":
    main()
