# BasicAgent 是助教提供的一个 Agent，原始代码请参考项目根目录 agents/basic_agent.py，下面代码已经经过我的调整，更加符合我们训练框架的使用

from typing import Any, Callable, Dict, List, Optional, override

import numpy as np
import pooltool as pt
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from src.utils.logger import get_logger

from .base import (
    ActionDict,
    Agent,
    BallsDict,
    analyze_shot_for_reward,
    simulate_with_timeout,
)

# Initialize logger
logger = get_logger()


class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""

    def __init__(
        self,
        target_balls=None,
        initial_search: int = 20,
        opt_search: int = 10,
        enable_noise: bool = False,
        noise_std: Optional[Dict[str, float]] = None,
    ):
        """初始化 Agent

        参数：
            target_balls: 保留参数，暂未使用
            initial_search: 初始随机采样点数
            opt_search: 后续优化迭代次数
            enable_noise: 是否启用噪声
            noise_std: 各参数的噪声标准差
        """
        super().__init__()

        # 搜索空间
        self.pbounds = {
            "V0": (0.5, 8.0),
            "phi": (0, 360),
            "theta": (0, 90),
            "a": (-0.5, 0.5),
            "b": (-0.5, 0.5),
        }

        # 优化参数 (now configurable via constructor)
        self.INITIAL_SEARCH = initial_search
        self.OPT_SEARCH = opt_search
        self.alpha = 1e-2

        # Bayes Optimizer settings
        self.n_restarts_optimizer = 10
        self.matern_nu = 2.5
        self.gamma_osc = 0.8
        self.gamma_pan = 1.0

        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = noise_std or {
            "V0": 0.1,
            "phi": 0.1,
            "theta": 0.1,
            "a": 0.003,
            "b": 0.003,
        }
        self.enable_noise = enable_noise

        logger.info("BasicAgent (贝叶斯优化版) 已初始化。")

    def _create_optimizer(self, reward_function: Callable, seed: int):
        """创建贝叶斯优化器

        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子

        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=self.matern_nu),
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=seed,
        )

        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=self.gamma_osc, gamma_pan=self.gamma_pan
        )

        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,  # TODO: 这个参数干什么的，默认是2？
            bounds_transformer=bounds_transformer,
        )
        optimizer._gp = gpr

        return optimizer

    @override
    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[Any] = None,
    ) -> ActionDict:
        """使用贝叶斯优化搜索最佳击球参数

        Args:
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        Returns:
            ActionDict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # Step 0: 输入验证
        if balls is None:
            logger.warning("[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        if my_targets is None:
            logger.warning("BasicAgent 未检测到目标击球，使用随机动作")
            return self._random_action()

        try:
            # Step 1: 状态快照 + 目标检查
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: ball.copy() for bid, ball in balls.items()}

            my_targets, switched = self.get_remaining_targets(balls, my_targets)
            if switched:
                logger.info("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建"奖励函数" (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: ball.copy() for bid, ball in balls.items()}
                sim_table = table.copy()
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std["V0"])
                        phi_noisy = phi + np.random.normal(0, self.noise_std["phi"])
                        theta_noisy = theta + np.random.normal(0, self.noise_std["theta"])
                        a_noisy = a + np.random.normal(0, self.noise_std["a"])
                        b_noisy = b + np.random.normal(0, self.noise_std["b"])

                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)

                        shot.cue.set_state(
                            V0=V0_noisy,
                            phi=phi_noisy,
                            theta=theta_noisy,
                            a=a_noisy,
                            b=b_noisy,
                        )
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)

                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception:
                    # 模拟失败，给予极大惩罚
                    return -500

                # 使用我们的"裁判"来打分
                score = analyze_shot_for_reward(
                    shot=shot, last_state=last_state_snapshot, player_targets=my_targets
                )

                return score

            logger.info(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")

            seed = np.random.randint(int(1e6))
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(init_points=self.INITIAL_SEARCH, n_iter=self.OPT_SEARCH)

            best_result = optimizer.max

            if best_result is None:
                raise ValueError(
                    "贝叶斯优化未返回任何结果"
                )  # This should not happen normally. It is handled by the outer try-except.
            best_params = best_result["params"]
            best_score = best_result["target"]

            if best_score < 10:
                logger.info(
                    f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。"
                )
                return self._random_action()
            action = {
                "V0": float(best_params["V0"]),
                "phi": float(best_params["phi"]),
                "theta": float(best_params["theta"]),
                "a": float(best_params["a"]),
                "b": float(best_params["b"]),
            }

            logger.info(
                f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}"
            )
            return action

        except Exception as e:
            logger.error(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback

            traceback.print_exc()
            return self._random_action()
