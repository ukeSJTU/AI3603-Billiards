#!/usr/bin/env python3
"""
train_ppo.py - PPO Agent 训练脚本

用法：
    python train_ppo.py --timesteps 200000 --save-freq 10000
    python train_ppo.py --timesteps 100000 --save-dir models/ppo_v2
    python train_ppo.py --help
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent import BasicAgent
from gym_wrapper import PoolGymEnv

# ============ Logger 导入（带降级） ============
try:
    from logger import get_logger, setup_logger

    setup_logger()
    log = get_logger(__name__)
except ImportError:

    class _FakeLogger:
        def info(self, msg):
            print(f"[INFO] {msg}")

        def warning(self, msg):
            print(f"[WARNING] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

    log = _FakeLogger()


def make_env():
    """创建环境工厂函数"""

    def _init():
        return PoolGymEnv(opponent_agent=BasicAgent())

    return _init


def train(
    total_timesteps: int = 200000,
    save_dir: str = "models",
    save_freq: int = 10000,
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
):
    """
    训练 PPO Agent

    参数：
        total_timesteps: 总训练步数（推荐 100k-500k）
        save_dir: 模型保存目录
        save_freq: 保存检查点频率
        eval_freq: 评估频率
        n_eval_episodes: 每次评估的对局数
    """
    log.info(f"开始训练 PPO Agent")
    log.info(f"总训练步数: {total_timesteps}")
    log.info(f"保存目录: {save_dir}")

    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"保存目录已创建: {save_dir}")

    # 创建训练环境（向量化）
    log.info("创建训练环境...")
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 创建评估环境
    log.info("创建评估环境...")
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # 创建 PPO 模型
    log.info("创建 PPO 模型...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{save_dir}/tensorboard/",
    )
    log.info("PPO 模型已创建")

    # 回调函数
    log.info("设置回调函数...")
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="ppo_checkpoint",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=f"{save_dir}/eval_logs",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # 开始训练
    log.info("=" * 60)
    log.info(f"开始训练 PPO Agent，总步数: {total_timesteps}")
    log.info(f"检查点保存频率: 每 {save_freq} 步")
    log.info(f"评估频率: 每 {eval_freq} 步（{n_eval_episodes} 局）")
    log.info(f"TensorBoard 日志: {save_dir}/tensorboard/")
    log.info("=" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
        )
    except KeyboardInterrupt:
        log.warning("训练被用户中断")

    # 保存最终模型
    log.info("保存最终模型...")
    model.save(f"{save_dir}/ppo_final")
    env.save(f"{save_dir}/vec_normalize_final.pkl")

    log.info("=" * 60)
    log.info(f"训练完成！")
    log.info(f"最终模型: {save_dir}/ppo_final.zip")
    log.info(f"最佳模型: {save_dir}/best_model.zip")
    log.info(f"归一化参数: {save_dir}/vec_normalize_final.pkl")
    log.info(f"查看训练曲线: tensorboard --logdir={save_dir}/tensorboard/")
    log.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="训练 PPO Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="总训练步数（推荐: 100k-500k）",
    )
    parser.add_argument(
        "--save-dir", type=str, default="models", help="模型保存目录"
    )
    parser.add_argument(
        "--save-freq", type=int, default=10000, help="检查点保存频率（步数）"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5000, help="评估频率（步数）"
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=5, help="每次评估的对局数"
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
