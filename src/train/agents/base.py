import random
import signal
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pooltool as pt
from pooltool import Ball

from src.utils.logger import get_logger

ActionDict = Dict[str, float]  # {'V0', 'phi', 'theta', 'a', 'b'}


class ActionBoundsDict(TypedDict):
    V0: Tuple[float, float]
    phi: Tuple[float, float]
    theta: Tuple[float, float]
    a: Tuple[float, float]
    b: Tuple[float, float]


BallsDict = Dict[str, Ball]  # {ball_id: Ball}

logger = get_logger()


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
        logger.warning(f"物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ 击球评分函数 ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）

    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']

    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）

    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """

    # 1. 基本分析
    new_pocketed = [
        bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4
    ]

    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [
        bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]
    ]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
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
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ["8"]:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
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
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ["8"]:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负

    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30

    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    # 合法无进球小奖励
    if (
        score == 0
        and not cue_pocketed
        and not eight_pocketed
        and not foul_first_hit
        and not foul_no_rail
    ):
        score = 10

    return score


# ============ Agent 抽象基类 ============
class Agent(ABC):
    """
    Agent 抽象基类

    设计原则：
    1. 定义统一的决策接口
    2. 提供通用工具方法（噪声、随机动作等）
    3. 子类只需实现 decision 方法
    """

    def __init__(self):
        """初始化基类"""
        self._action_bounds: ActionBoundsDict = {
            "V0": (5.0, 30.0),  # 初速度范围（单位：m/s）
            "phi": (0.0, 360.0),  # 水平角度范围（单位：度）
            "theta": (0.0, 45.0),  # 垂直角度范围（单位：度）
            "a": (-1.0, 1.0),  # 横向旋转范围
            "b": (-1.0, 1.0),  # 纵向旋转范围
        }

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

    def _random_action(self) -> ActionDict:
        """
        生成随机击球动作

        Returns:
            ActionDict: 在动作空间边界内的随机动作
        """
        bounds = self._action_bounds
        return {
            "V0": round(random.uniform(*bounds["V0"]), 2),
            "phi": round(random.uniform(*bounds["phi"]), 2),
            "theta": round(random.uniform(*bounds["theta"]), 2),
            "a": round(random.uniform(*bounds["a"]), 3),
            "b": round(random.uniform(*bounds["b"]), 3),
        }

    @staticmethod
    def get_remaining_targets(
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

    # Alias for backward compatibility
    check_remaining_targets = get_remaining_targets
