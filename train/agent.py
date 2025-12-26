"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- BaseAgent: 抽象基类，定义决策接口和公共方法
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数

设计说明：
- 使用 ABC 定义抽象接口（train 环境）
- 提供 ABC 兼容层，方便代码粘贴到 eval（eval 可能没有 ABC 依赖）
"""

import copy
import random
import signal

# ============ ABC 兼容层 ============
# 设计目的: 在 train 中使用 ABC 强制接口约束，
# 但代码粘贴到 eval 时可以无缝降级为普通类
# try:
#     from abc import ABC, abstractmethod
#     _HAS_ABC = True
# except ImportError:
#     _HAS_ABC = False
#     class ABC:  # type: ignore
#         """ABC 占位符（降级模式）"""
#         pass
#     def abstractmethod(func):  # type: ignore
#         """abstractmethod 占位符（降级模式）"""
#         return func
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pooltool as pt

# ============ 第三方库导入 ============
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 配置导入（带降级） ============
try:
    from config import (
        ACTION_BOUNDS,
        DEFAULT_BAYES_CONFIG,
        DEFAULT_EVAL_CONFIG,
        DEFAULT_NOISE_CONFIG,
        REWARD_CONFIG,
    )

    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False
    # 降级: 内联默认值（用于 eval 环境）
    # 这些值会在类内部定义


# ============ Logger 导入（带降级） ============
try:
    from logger import get_logger

    log = get_logger(__name__)
except ImportError:
    # 降级: 使用 print 作为 fallback

    class _FakeLogger:
        """简单的 print-based logger（降级模式）"""

        def info(self, msg):
            print(f"[INFO] {msg}")

        def debug(self, msg):
            pass  # DEBUG 级别在降级模式下不输出

        def warning(self, msg):
            print(f"[WARNING] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

    log = _FakeLogger()


# ============ 类型别名 ============
ActionDict = Dict[str, float]  # {'V0', 'phi', 'theta', 'a', 'b'}
BallsDict = Dict[str, Any]  # {ball_id: Ball}


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""

    pass


def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")


def simulate_with_timeout(shot, timeout: int = 3) -> bool:
    """带超时保护的物理模拟

    Args:
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒

    Returns:
        bool: True 表示模拟成功，False 表示超时或失败

    Note:
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        Windows 系统或非主线程调用可能无法正常工作
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        log.warning(f"物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ 击球评分函数 ============
def analyze_shot_for_reward(
    shot: pt.System, last_state: dict, player_targets: list
) -> float:
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）

    Args:
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']

    Returns:
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）

    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    # 获取奖励配置
    if _HAS_CONFIG:
        rewards = REWARD_CONFIG
    else:
        # 内联默认值
        class rewards:  # type: ignore
            own_ball_pocketed = 50
            legal_eight_ball = 100
            legal_no_pocket = 10
            cue_pocketed = -100
            illegal_eight = -150
            cue_and_eight = -150
            foul_first_hit = -30
            foul_no_rail = -30
            enemy_pocketed = -20

    # 1. 基本分析 - 找出新进袋的球
    new_pocketed = [
        bid
        for bid, b in shot.balls.items()
        if b.state.s == 4 and last_state[bid].state.s != 4
    ]

    # 根据 player_targets 判断进球归属
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [
        bid
        for bid in new_pocketed
        if bid not in player_targets and bid not in ["cue", "8"]
    ]
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    }

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    # 首球犯规判定
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ["8"]:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if "cushion" in et:
            if "cue" in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if (
        len(new_pocketed) == 0
        and first_contact_ball_id is not None
        and (not cue_hit_cushion)
        and (not target_hit_cushion)
    ):
        foul_no_rail = True

    # 4. 计算奖励分数
    score = 0

    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score += rewards.cue_and_eight
    elif cue_pocketed:
        score += rewards.cue_pocketed
    elif eight_pocketed:
        if player_targets == ["8"]:
            score += rewards.legal_eight_ball
        else:
            score += rewards.illegal_eight

    # 首球犯规和碰库犯规
    if foul_first_hit:
        score += rewards.foul_first_hit
    if foul_no_rail:
        score += rewards.foul_no_rail

    # 进球得分
    score += len(own_pocketed) * rewards.own_ball_pocketed
    score += len(enemy_pocketed) * rewards.enemy_pocketed

    # 合法无进球小奖励
    if (
        score == 0
        and not cue_pocketed
        and not eight_pocketed
        and not foul_first_hit
        and not foul_no_rail
    ):
        score = rewards.legal_no_pocket

    return score


# ============ Agent 抽象基类 ============
class BaseAgent(ABC):
    """
    Agent 抽象基类

    设计原则：
    1. 定义统一的决策接口
    2. 提供通用工具方法（噪声、随机动作等）
    3. 子类只需实现 decision 方法
    """

    def __init__(self):
        """初始化基类"""
        if _HAS_CONFIG:
            self._action_bounds = ACTION_BOUNDS
        else:
            self._action_bounds = self._default_bounds()

    @staticmethod
    def _default_bounds():
        """内联默认边界（用于 eval 降级）"""

        class _Bounds:
            V0 = (0.5, 8.0)
            phi = (0.0, 360.0)
            theta = (0.0, 90.0)
            a = (-0.5, 0.5)
            b = (-0.5, 0.5)

            def as_dict(self):
                return {
                    "V0": self.V0,
                    "phi": self.phi,
                    "theta": self.theta,
                    "a": self.a,
                    "b": self.b,
                }

        return _Bounds()

    @abstractmethod
    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[Any] = None,
    ) -> ActionDict:
        """
        决策方法（子类必须实现）

        Args:
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        Returns:
            ActionDict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        raise NotImplementedError

    def random_action(self) -> ActionDict:
        """
        生成随机击球动作

        Returns:
            ActionDict: 在动作空间边界内的随机动作
        """
        bounds = self._action_bounds
        return {
            "V0": round(random.uniform(*bounds.V0), 2),
            "phi": round(random.uniform(*bounds.phi), 2),
            "theta": round(random.uniform(*bounds.theta), 2),
            "a": round(random.uniform(*bounds.a), 3),
            "b": round(random.uniform(*bounds.b), 3),
        }

    def _random_action(self) -> ActionDict:
        """兼容旧代码（已废弃，请使用 random_action）"""
        return self.random_action()

    @staticmethod
    def apply_noise(
        action: ActionDict,
        noise_std: Dict[str, float],
        bounds: Optional[Any] = None,
    ) -> ActionDict:
        """
        对动作添加高斯噪声

        Args:
            action: 原始动作
            noise_std: 各参数的噪声标准差
            bounds: 动作边界（用于裁剪）

        Returns:
            ActionDict: 添加噪声后的动作
        """
        if bounds is None:
            if _HAS_CONFIG:
                bounds = ACTION_BOUNDS
            else:
                bounds = BaseAgent._default_bounds()

        noisy = {}
        for key, val in action.items():
            noisy_val = val + np.random.normal(0, noise_std.get(key, 0))
            bound = getattr(bounds, key, (-np.inf, np.inf))
            if key == "phi":
                noisy_val = noisy_val % 360
            else:
                noisy_val = np.clip(noisy_val, *bound)
            noisy[key] = float(noisy_val)
        return noisy

    @staticmethod
    def check_remaining_targets(
        balls: BallsDict,
        my_targets: List[str],
    ) -> Tuple[List[str], bool]:
        """
        检查剩余目标球并判断是否需要切换到黑8

        Args:
            balls: 球状态字典
            my_targets: 当前目标球列表

        Returns:
            Tuple[List[str], bool]: (更新后的目标球列表, 是否切换到黑8)
        """
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            return ["8"], True
        return my_targets, False


# ============ 兼容别名 ============
Agent = BaseAgent


# ============ BasicAgent 实现 ============
class BasicAgent(BaseAgent):
    """基于贝叶斯优化的智能 Agent"""

    def __init__(self, target_balls=None):
        """初始化 Agent

        Args:
            target_balls: 保留参数，暂未使用
        """
        super().__init__()

        # 加载配置
        if _HAS_CONFIG:
            self._bayes_config = DEFAULT_BAYES_CONFIG
            self._noise_config = DEFAULT_NOISE_CONFIG
            self._eval_config = DEFAULT_EVAL_CONFIG
        else:
            self._bayes_config = self._default_bayes_config()
            self._noise_config = self._default_noise_config()
            self._eval_config = self._default_eval_config()

        log.info("BasicAgent (Smart, pooltool-native) 已初始化。")

    @staticmethod
    def _default_bayes_config():
        """内联默认贝叶斯配置"""

        class _Config:
            initial_search = 20
            opt_search = 10
            alpha = 1e-2
            gamma_osc = 0.8
            gamma_pan = 1.0
            n_restarts_optimizer = 10
            matern_nu = 2.5

        return _Config()

    @staticmethod
    def _default_noise_config():
        """内联默认噪声配置"""

        class _Config:
            enabled = False

            def as_dict(self):
                return {
                    "V0": 0.1,
                    "phi": 0.1,
                    "theta": 0.1,
                    "a": 0.003,
                    "b": 0.003,
                }

        return _Config()

    @staticmethod
    def _default_eval_config():
        """内联默认评估配置"""

        class _Config:
            simulation_timeout = 3
            min_score_threshold = 10.0

        return _Config()

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
        if balls is None or my_targets is None:
            log.warning("Agent decision函数未收到关键信息，使用随机动作。")
            return self.random_action()

        try:
            # Step 1: 状态快照 + 目标检查
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            my_targets, switched = self.check_remaining_targets(balls, my_targets)
            if switched:
                log.info("我的目标球已全部清空，自动切换目标为：8号球")

            # Step 2: 构建奖励函数
            def reward_fn(V0, phi, theta, a, b):
                return self._evaluate_shot(
                    V0, phi, theta, a, b, balls, table, last_state, my_targets
                )

            # Step 3: 运行贝叶斯优化
            log.info(f"正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            seed = np.random.randint(int(1e6))
            optimizer = self._create_optimizer(reward_fn, seed)
            optimizer.maximize(
                init_points=self._bayes_config.initial_search,
                n_iter=self._bayes_config.opt_search,
            )

            # Step 4: 提取最优结果
            best = optimizer.max
            if best is None:
                log.warning("贝叶斯优化未找到结果，使用随机动作。")
                return self.random_action()

            best_params = best["params"]
            best_score = best["target"]

            # Step 5: 低分兜底
            if best_score < self._eval_config.min_score_threshold:
                log.info(f"未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self.random_action()

            action = {
                "V0": float(best_params["V0"]),
                "phi": float(best_params["phi"]),
                "theta": float(best_params["theta"]),
                "a": float(best_params["a"]),
                "b": float(best_params["b"]),
            }

            log.info(
                f"决策 (得分: {best_score:.2f}): "
                f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                f"theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}"
            )
            return action

        except Exception as e:
            log.error(f"决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback

            traceback.print_exc()
            return self.random_action()

    def _create_optimizer(self, reward_function, seed: int):
        """创建贝叶斯优化器

        Args:
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子

        Returns:
            BayesianOptimization 对象
        """
        cfg = self._bayes_config

        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=cfg.matern_nu),
            alpha=cfg.alpha,
            n_restarts_optimizer=cfg.n_restarts_optimizer,
            random_state=seed,
        )

        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=cfg.gamma_osc,
            gamma_pan=cfg.gamma_pan,
        )

        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self._action_bounds.as_dict(),
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer,
        )
        optimizer._gp = gpr

        return optimizer

    def _evaluate_shot(
        self,
        V0: float,
        phi: float,
        theta: float,
        a: float,
        b: float,
        balls: BallsDict,
        table: Any,
        last_state: BallsDict,
        my_targets: List[str],
    ) -> float:
        """
        评估单次击球的奖励分数

        Args:
            V0, phi, theta, a, b: 击球参数
            balls: 当前球状态
            table: 球桌对象
            last_state: 击球前状态快照
            my_targets: 目标球列表

        Returns:
            float: 奖励分数
        """
        # 创建沙盒环境
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        try:
            # 构建动作
            action = {"V0": V0, "phi": phi, "theta": theta, "a": a, "b": b}

            # 应用噪声（如果启用）
            if self._noise_config.enabled:
                action = self.apply_noise(action, self._noise_config.as_dict())

            shot.cue.set_state(**action)

            # 带超时的模拟
            if not simulate_with_timeout(
                shot, timeout=self._eval_config.simulation_timeout
            ):
                return 0  # 超时是物理引擎问题，不惩罚 agent

        except Exception:
            return -500

        return analyze_shot_for_reward(shot, last_state, my_targets)


# ============ NewAgent 实现 ============
class NewAgent(BaseAgent):
    """
    基于几何启发 + 多袋口评估 + 精细搜索的智能 Agent

    策略：
    1. 对每个目标球计算到所有6个袋口的可行性
    2. 检查路径上是否有其他球阻挡
    3. 选择最容易进袋的球-袋组合
    4. 在最优几何角度附近做密集采样+模拟验证
    """

    # 袋口ID列表
    POCKET_IDS = ["lb", "lc", "lt", "rb", "rc", "rt"]
    # 球的半径（米），标准台球
    BALL_RADIUS = 0.028575

    def __init__(self):
        super().__init__()
        # 搜索参数
        self.n_candidates = 30  # 候选动作数量
        self.angle_spread = 5.0  # 角度搜索范围（度）
        self.v0_spread = 1.0  # 速度搜索范围
        log.info("NewAgent (Geometry-Guided) 已初始化。")

    # ============ 几何工具函数 ============
    @staticmethod
    def get_ball_pos(ball) -> np.ndarray:
        """获取球的2D位置 (x, y)"""
        return ball.state.rvw[0][:2]

    @staticmethod
    def get_pocket_pos(table, pocket_id: str) -> np.ndarray:
        """获取袋口的2D位置 (x, y)"""
        return table.pockets[pocket_id].center[:2]

    @staticmethod
    def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """计算两点间距离"""
        return float(np.linalg.norm(p2 - p1))

    @staticmethod
    def calc_angle(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        """
        计算从 from_pos 到 to_pos 的角度（度）
        返回值范围：[0, 360)
        """
        delta = to_pos - from_pos
        angle_rad = np.arctan2(delta[1], delta[0])
        angle_deg = np.degrees(angle_rad)
        return angle_deg % 360

    def is_path_blocked(
        self,
        start: np.ndarray,
        end: np.ndarray,
        balls: BallsDict,
        exclude_ids: List[str],
    ) -> bool:
        """
        检查从 start 到 end 的路径是否被其他球阻挡

        Args:
            start: 起点位置 (x, y)
            end: 终点位置 (x, y)
            balls: 所有球
            exclude_ids: 排除的球ID（如白球、目标球本身）

        Returns:
            bool: True 表示路径被阻挡
        """
        direction = end - start
        path_length = np.linalg.norm(direction)
        if path_length < 1e-6:
            return False
        direction = direction / path_length  # 单位向量

        for bid, ball in balls.items():
            if bid in exclude_ids:
                continue
            if ball.state.s == 4:  # 已进袋，跳过
                continue

            ball_pos = self.get_ball_pos(ball)
            # 计算球心到路径直线的距离
            # 使用向量投影
            to_ball = ball_pos - start
            proj_length = np.dot(to_ball, direction)

            # 球必须在路径上（不在起点前或终点后）
            if proj_length < self.BALL_RADIUS or proj_length > path_length - self.BALL_RADIUS:
                continue

            # 计算垂直距离
            proj_point = start + proj_length * direction
            perp_dist = np.linalg.norm(ball_pos - proj_point)

            # 如果距离小于两个球的直径，则阻挡
            if perp_dist < 2 * self.BALL_RADIUS + 0.005:  # 加一点余量
                return True

        return False

    def evaluate_ball_pocket_pair(
        self,
        cue_pos: np.ndarray,
        target_ball_id: str,
        target_pos: np.ndarray,
        pocket_pos: np.ndarray,
        balls: BallsDict,
    ) -> Tuple[float, float, float]:
        """
        评估白球-目标球-袋口组合的可行性

        Returns:
            Tuple[float, float, float]: (分数, 理想phi角度, 理想V0)
            分数越高越好，-1表示不可行
        """
        # 1. 检查目标球到袋口的路径是否被阻挡
        if self.is_path_blocked(target_pos, pocket_pos, balls, ["cue", target_ball_id]):
            return -1, 0, 0

        # 2. 检查白球到目标球的路径是否被阻挡
        if self.is_path_blocked(cue_pos, target_pos, balls, ["cue", target_ball_id]):
            return -1, 0, 0

        # 3. 计算几何角度
        # 目标球到袋口的角度
        target_to_pocket_angle = self.calc_angle(target_pos, pocket_pos)
        # 理想击球点：目标球背面（相对于袋口）
        # 白球需要打到目标球上，使其朝袋口方向运动
        # 理想击球角度 = 从白球到目标球的角度，使得碰撞后目标球朝袋口
        # 简化：白球应该打向目标球的"背面"
        ideal_phi = target_to_pocket_angle  # 白球击向目标球，期望目标球朝袋口

        # 4. 计算实际需要的 phi（白球到目标球的角度）
        cue_to_target_angle = self.calc_angle(cue_pos, target_pos)

        # 5. 计算角度偏差（白球击球方向与理想方向的差异）
        # 理想情况：白球、目标球、袋口三点共线
        angle_diff = abs(cue_to_target_angle - ideal_phi)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # 角度偏差过大（>70度）基本不可能打进
        if angle_diff > 70:
            return -1, 0, 0

        # 6. 计算距离
        cue_to_target_dist = self.calc_distance(cue_pos, target_pos)
        target_to_pocket_dist = self.calc_distance(target_pos, pocket_pos)
        total_dist = cue_to_target_dist + target_to_pocket_dist

        # 7. 计算分数
        # 分数考虑：角度偏差（越小越好）、距离（越近越好）
        angle_score = max(0, 100 - angle_diff * 2)  # 角度偏差惩罚
        dist_score = max(0, 100 - total_dist * 30)  # 距离惩罚
        score = angle_score * 0.7 + dist_score * 0.3

        # 8. 估算理想速度
        # 简单估算：根据总距离
        ideal_v0 = np.clip(1.5 + total_dist * 2.0, 1.0, 6.0)

        return score, cue_to_target_angle, ideal_v0

    def find_best_shot(
        self,
        balls: BallsDict,
        my_targets: List[str],
        table: Any,
    ) -> Tuple[Optional[str], Optional[str], float, float]:
        """
        找到最佳的球-袋组合

        Returns:
            Tuple[Optional[str], Optional[str], float, float]: (目标球ID, 袋口ID, 理想phi, 理想V0)
            如果没有可行方案，返回 (None, None, 0, 0)
        """
        cue_pos = self.get_ball_pos(balls["cue"])

        best_score = -1
        best_target = None
        best_pocket = None
        best_phi = 0
        best_v0 = 2.0

        # 遍历所有目标球
        for target_id in my_targets:
            if balls[target_id].state.s == 4:  # 已进袋
                continue

            target_pos = self.get_ball_pos(balls[target_id])

            # 遍历所有袋口
            for pocket_id in self.POCKET_IDS:
                pocket_pos = self.get_pocket_pos(table, pocket_id)

                score, phi, v0 = self.evaluate_ball_pocket_pair(
                    cue_pos, target_id, target_pos, pocket_pos, balls
                )

                if score > best_score:
                    best_score = score
                    best_target = target_id
                    best_pocket = pocket_id
                    best_phi = phi
                    best_v0 = v0

        if best_score < 0:
            return None, None, 0, 0

        log.debug(
            f"最佳方案: 球{best_target} -> 袋{best_pocket}, "
            f"分数={best_score:.1f}, phi={best_phi:.1f}, V0={best_v0:.1f}"
        )
        return best_target, best_pocket, best_phi, best_v0

    def generate_candidates(
        self,
        base_phi: float,
        base_v0: float,
    ) -> List[ActionDict]:
        """
        在基础参数附近生成候选动作
        """
        candidates = []

        for _ in range(self.n_candidates):
            phi = base_phi + np.random.uniform(-self.angle_spread, self.angle_spread)
            phi = phi % 360

            v0 = base_v0 + np.random.uniform(-self.v0_spread, self.v0_spread)
            v0 = np.clip(v0, 0.8, 7.0)

            # theta, a, b 使用小范围随机
            theta = np.random.uniform(0, 15)  # 小角度
            a = np.random.uniform(-0.2, 0.2)
            b = np.random.uniform(-0.2, 0.2)

            candidates.append({
                "V0": float(v0),
                "phi": float(phi),
                "theta": float(theta),
                "a": float(a),
                "b": float(b),
            })

        # 添加基础动作（无扰动）
        candidates.append({
            "V0": float(base_v0),
            "phi": float(base_phi),
            "theta": 2.0,
            "a": 0.0,
            "b": 0.0,
        })

        return candidates

    def evaluate_candidate(
        self,
        action: ActionDict,
        balls: BallsDict,
        table: Any,
        last_state: BallsDict,
        my_targets: List[str],
    ) -> float:
        """评估单个候选动作的得分"""
        # 创建沙盒环境
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        try:
            shot.cue.set_state(**action)
            if not simulate_with_timeout(shot, timeout=3):
                return -100  # 超时
        except Exception:
            return -500

        return analyze_shot_for_reward(shot, last_state, my_targets)

    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[Any] = None,
    ) -> ActionDict:
        """
        决策方法：几何启发 + 多袋口评估 + 精细搜索

        Args:
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        Returns:
            ActionDict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # Step 0: 输入验证
        if balls is None or my_targets is None or table is None:
            log.warning("NewAgent: 输入不完整，使用随机动作")
            return self.random_action()

        try:
            # Step 1: 状态快照 + 目标检查
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            my_targets, switched = self.check_remaining_targets(balls, my_targets)
            if switched:
                log.info("目标球已清空，切换到8号球")

            # Step 2: 几何分析找最佳球-袋组合
            best_target, best_pocket, base_phi, base_v0 = self.find_best_shot(
                balls, my_targets, table
            )

            if best_target is None:
                # 没有明显好的方案，扩大搜索
                log.info("无明显最佳方案，使用扩大搜索")
                base_phi = np.random.uniform(0, 360)
                base_v0 = np.random.uniform(2.0, 5.0)
                self.n_candidates = 50  # 临时增加候选数
            else:
                log.info(f"几何分析: 目标球{best_target} -> 袋口{best_pocket}")

            # Step 3: 生成候选动作
            candidates = self.generate_candidates(base_phi, base_v0)

            # Step 4: 评估所有候选
            best_action = None
            best_score = float("-inf")

            for action in candidates:
                score = self.evaluate_candidate(
                    action, balls, table, last_state, my_targets
                )
                if score > best_score:
                    best_score = score
                    best_action = action

            # Step 5: 结果处理
            if best_action is None or best_score < -50:
                log.info(f"未找到好方案 (最高分: {best_score:.1f})，使用随机动作")
                return self.random_action()

            log.info(
                f"决策 (得分: {best_score:.1f}): "
                f"V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}"
            )
            return best_action

        except Exception as e:
            log.error(f"决策出错: {e}")
            import traceback
            traceback.print_exc()
            return self.random_action()
