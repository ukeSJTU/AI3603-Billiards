"""
ppo_agent.py - 基于 Stable-Baselines3 的 PPO Agent

设计要点：
- 继承 BaseAgent，实现 decision() 接口
- 内部封装训练好的 SB3 模型
- 支持加载预训练模型
"""

import copy
from pathlib import Path
from typing import Any, List, Optional

from gym_wrapper import PoolGymEnv
from stable_baselines3 import PPO

from agent import ActionDict, BallsDict, BaseAgent

# ============ Logger 导入（带降级） ============
try:
    from logger import get_logger

    log = get_logger(__name__)
except ImportError:

    class _FakeLogger:
        """简单的 print-based logger（降级模式）"""

        def info(self, msg):
            print(f"[INFO] {msg}")

        def debug(self, msg):
            pass

        def warning(self, msg):
            print(f"[WARNING] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

    log = _FakeLogger()


class PPOAgent(BaseAgent):
    """
    基于 PPO 的强化学习 Agent

    用法：
        # 使用默认最佳模型
        agent = PPOAgent()

        # 使用指定模型
        agent = PPOAgent(model_path="models/ppo_final.zip")

        # 使用检查点模型
        agent = PPOAgent(model_path="models/ppo_checkpoint_200000_steps.zip")
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__()

        # 默认使用最佳模型
        if model_path is None:
            model_path = "models/best_model.zip"

        self.model_path = model_path
        self.model = None
        self.env = None

        if model_path and Path(model_path).exists():
            self.load(model_path)
            log.info(f"PPOAgent 加载模型: {model_path}")
        else:
            if model_path:
                log.warning(f"PPOAgent: 模型路径 {model_path} 不存在")
            log.info("PPOAgent 未加载模型（训练模式或推理将使用随机动作）")

    def load(self, model_path: str):
        """加载训练好的模型"""
        self.model = PPO.load(model_path)
        self.model_path = model_path
        log.info(f"PPOAgent: 成功加载模型 {model_path}")

    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[Any] = None,
    ) -> ActionDict:
        """
        决策方法（继承自 BaseAgent）

        将 PoolEnv 观测转为 Gym 观测，然后用 PPO 模型预测
        """
        if self.model is None:
            log.warning("PPOAgent: 模型未加载，使用随机动作")
            return self.random_action()

        if balls is None or my_targets is None or table is None:
            log.warning("PPOAgent: 输入不完整，使用随机动作")
            return self.random_action()

        try:
            # 创建临时环境用于观测转换
            # 注意：这里是 hack，更优雅的方式是缓存环境
            if self.env is None:
                self.env = PoolGymEnv()

            # 手动设置环境状态（模拟 reset 后的状态）
            self.env.env.balls = copy.deepcopy(balls)
            self.env.env.player_targets["A"] = my_targets
            self.env.env.table = copy.deepcopy(table)

            # 获取观测
            obs = self.env._get_observation()

            # PPO 预测
            action, _states = self.model.predict(obs, deterministic=True)

            # 转为字典格式
            action_dict = {
                "V0": float(action[0]),
                "phi": float(action[1]),
                "theta": float(action[2]),
                "a": float(action[3]),
                "b": float(action[4]),
            }

            return action_dict

        except Exception as e:
            log.error(f"PPOAgent 决策出错: {e}")
            import traceback

            traceback.print_exc()
            return self.random_action()
