"""
poolenv.py - 台球环境模块

实现八球台球对战环境：
- PoolEnv: 双人对战环境类，管理游戏状态和规则
- collect_ball_states: 收集球状态
- save_balls_state / restore_balls_state: 状态保存/恢复

主要接口：
- reset(): 初始化游戏
- get_observation(): 获取观测
- take_shot(action): 执行击球
- get_done(): 检查游戏结束
"""

import copy

import numpy as np
import pooltool as pt

from src.train.agents import ActionDict, BallsDict
from src.utils.logger import get_logger

logger = get_logger()


def collect_ball_states(shot):
    """收集球状态信息

    参数：
        shot: System 对象

    返回：
        dict: {ball_id: {'position', 'velocity', 'spin', 'state', 'time', 'pocketed'}}
    """
    results = {}
    for ball_id, ball in shot.balls.items():
        s = ball.state
        results[ball_id] = {
            "position": s.rvw[0].tolist(),
            "velocity": s.rvw[1].tolist(),
            "spin": s.rvw[2].tolist(),
            "state": int(s.s),
            "time": float(s.t),
            "pocketed": ball.state.s,
        }
    return results


def save_balls_state(balls: BallsDict) -> dict:
    """保存球状态（深拷贝）

    参数：
        balls: {ball_id: Ball}

    返回：
        dict: 球状态副本
    """
    return {bid: copy.deepcopy(ball) for bid, ball in balls.items()}


def restore_balls_state(saved_state):
    """恢复球状态（深拷贝）

    参数：
        saved_state: 保存的球状态

    返回：
        dict: 恢复的球状态副本
    """
    return {bid: copy.deepcopy(ball) for bid, ball in saved_state.items()}


class PoolEnv:
    """台球对战环境"""

    def __init__(self):
        """初始化环境（需调用 reset() 后才能使用）"""
        # 桌面和球
        self.table = None
        self.balls = None
        self.cue = None

        # A和B方的球的ID
        # NOTE: 数据结构：{'A': [球ID列表], 'B': [球ID列表]}
        self.player_targets = None
        # 击球数
        self.hit_count = 0
        # 上一时刻的状态
        # NOTE: 用于犯规时回滚
        self.last_state = None
        # player的名称
        # NOTE: 0 represents 'A', 1 represents 'B'
        self.players = ["A", "B"]
        # 当前击球方
        self.curr_player = 0
        # 是否结束
        self.done = False
        # 赢家
        self.winner = None  # 'A', 'B', 'SAME'
        # 最大击球数
        self.MAX_HIT_COUNT = 60
        # 记录所有shot，用于赛后render正常比赛，或者保存比赛记录
        self.shot_record = pt.MultiSystem()

        # 击球参数噪声标准差（模拟真实误差）（前期调试的时候可以先禁用）（0.1-0.1-0.1-0.003-0.003这个组合就显著让agent的性能退化 从单局平均25杆到了单局平均35杆）
        self.noise_std = {
            "V0": 0.1,  # 速度标准差
            "phi": 0.1,  # 水平角度标准差（度）
            "theta": 0.1,  # 垂直角度标准差（度）
            "a": 0.003,  # 横向偏移标准差 球半径的比例（无量纲）
            "b": 0.003,  # 纵向偏移标准差 球半径的比例（无量纲）
        }
        self.enable_noise = True  # 是否启用噪声

    # NOTE: 环境给 agent 的观测是什么
    def get_observation(self, player=None):
        """
        功能：获取指定玩家的观测信息（深拷贝）

        输入参数：
            player (str, optional): 玩家标识，'A' 或 'B'
                若为 None，则返回当前击球方的观测

        返回值：
            tuple: (balls, my_targets, table)
            NOTE: 这个返回的就是 agent.decision() 函数需要的输入格式

                balls (dict): 球状态字典，{ball_id: Ball对象}
                    ball_id 取值：
                        - 'cue': 白球
                        - '1'-'7': 实心球（solid）
                        - '8': 黑8
                        - '9'-'15': 条纹球（stripe）

                    Ball 对象属性：
                        ball.state.rvw: np.ndarray, shape=(3,3)
                            [0]: position, np.array([x, y, z])  # 位置，单位：米
                            [1]: velocity, np.array([vx, vy, vz])  # 速度，单位：米/秒
                            [2]: spin, np.array([wx, wy, wz])  # 角速度，单位：弧度/秒

                        ball.state.s: int  # 状态码
                            0 = 静止状态
                            4 = 已进袋（通过 ball.state.s == 4 判断）
                            1-3 = 运动中间状态（滑动/滚动/旋转）

                        ball.state.t: float  # 时间戳，单位：秒

                    示例：
                        pos = balls['cue'].state.rvw[0]  # 白球位置
                        pocketed = (balls['1'].state.s == 4)  # 1号球是否进袋

                my_targets (list[str]): 该玩家的目标球ID列表
                    - 正常情况：['1', '2', ...] 或 ['9', '10', ...]
                    - 目标球全部进袋后：['8']（需打黑8）

                table (Table): 球桌对象
                    属性：
                        table.w: float  # 球桌宽度，单位：米（约0.99米）
                        table.l: float  # 球桌长度，单位：米（约1.98米）

                        table.pockets: dict, {pocket_id: Pocket对象}
                            pocket_id 取值：
                                'lb', 'lc', 'lt'  # 左侧：下、中、上
                                'rb', 'rc', 'rt'  # 右侧：下、中、上

                            Pocket.center: np.array([x, y, z])  # 球袋中心坐标

                        table.cushion_segments: CushionSegments  # 库边信息

                    示例：
                        width = table.w
                        lb_pos = table.pockets['lb'].center
                        pocket_ids = list(table.pockets.keys())
        """
        # 如果没给player信息，则默认给当前击球方的observation

        assert self.player_targets is not None, "环境未初始化，请先调用 reset() 方法！"

        if player is None:
            player = self.get_curr_player()
        # 返回当前所有球的信息，以及我方球的ID
        return (
            copy.deepcopy(self.balls),
            self.player_targets[player],
            copy.deepcopy(self.table),
        )

    def get_curr_player(
        self,
    ):
        """获取当前击球方

        返回：str, 'A' 或 'B'
        """
        return self.players[self.curr_player]

    def get_done(
        self,
    ):
        """检查游戏是否结束

        返回：tuple
            (True, {'winner': 'A'/'B'/'SAME', 'hit_count': int})  # 已结束
            (False, {})  # 未结束
        """
        if self.done:
            return True, {"winner": self.winner, "hit_count": self.hit_count}
        return False, {}

    # NOTE: 开局怎么摆球、怎么分配球型
    # ? 我不理解这个方法存在的意义，为什么不直接在 __init__ 里初始化？也许是觉得一次启动评测需要多局比赛？每一次比赛结束后都要 reset 一下？
    def reset(self, target_ball: str, state=None):
        """重置环境

        参数：
            state: 保留参数，必须为 None
            target_ball: Player A 目标球型
                'solid': A打实心(1-7), B打条纹(9-15)
                'stripe': A打条纹(9-15), B打实心(1-7)
        """
        # 目前不支持恢复到指定state，只能恢复到新开一局的状态
        # ? 那这个参数的意义在哪里？
        if state is not None:
            raise NotImplementedError("目前不支持恢复到指定state!")
        # 设置球场的初始状态
        self.table = pt.Table.default()
        self.balls = pt.get_rack(pt.GameType.EIGHTBALL, self.table)
        self.cue = pt.Cue(cue_ball_id="cue")
        # 设置player A 和 B 分别打什么类型的球
        if target_ball == "solid":
            self.player_targets = {
                "A": [str(i) for i in range(1, 8)],
                "B": [str(i) for i in range(9, 16)],
            }
        elif target_ball == "stripe":
            self.player_targets = {
                "A": [str(i) for i in range(9, 16)],
                "B": [str(i) for i in range(1, 8)],
            }
        else:
            raise NotImplementedError("不受支持的target_ball参数", target_ball)
        # 设置击球数为0
        self.hit_count = 0
        # 初始状态保存 (在第一次击球前，用作犯规回滚)
        self.last_state = save_balls_state(self.balls)
        # 设置两方player的名字
        # self.players = ["A", "B"]
        # 设置当前击球手为 player A
        self.curr_player = 0
        # 设置当前的done为False，且winner为None
        self.done = False
        self.winner = None
        # 清空记录所有shot的列表
        self.shot_record = pt.MultiSystem()

    # NOTE: 执行一杆击球动作
    def take_shot(self, action: ActionDict):
        """执行击球动作

        参数：
            action: {'V0': [0.5,8.0], 'phi': [0,360], 'theta': [0,90], 'a': [-0.5,0.5], 'b': [-0.5,0.5]}

        返回：dict
            必有字段：
                ME_INTO_POCKET: list[str]
                ENEMY_INTO_POCKET: list[str]
                WHITE_BALL_INTO_POCKET: bool
                BLACK_BALL_INTO_POCKET: bool
                BALLS: dict

            条件字段：
                FOUL_FIRST_HIT: bool  # 仅当 hit_count < MAX_HIT_COUNT
                NO_POCKET_NO_RAIL: bool  # 仅当 hit_count < MAX_HIT_COUNT
                NO_HIT: bool  # 仅当白球未接触任何球

        注：enable_noise=True 时添加高斯噪声
        """

        assert self.player_targets is not None, "环境未初始化，请先调用 reset() 方法！"
        assert self.table is not None, "环境未初始化，请先调用 reset() 方法！"
        assert self.last_state is not None, "环境未初始化，请先调用 reset() 方法！"
        assert self.cue is not None, "环境未初始化，请先调用 reset() 方法！"
        assert not self.done, "游戏已结束，无法继续击球！"

        # 添加高斯噪声模拟真实误差
        if self.enable_noise:
            noisy_action = {
                "V0": action["V0"] + np.random.normal(0, self.noise_std["V0"]),
                "phi": action["phi"] + np.random.normal(0, self.noise_std["phi"]),
                "theta": action["theta"] + np.random.normal(0, self.noise_std["theta"]),
                "a": action["a"] + np.random.normal(0, self.noise_std["a"]),
                "b": action["b"] + np.random.normal(0, self.noise_std["b"]),
            }

            # 限制参数在合理范围内
            noisy_action["V0"] = np.clip(noisy_action["V0"], 0.5, 8.0)
            noisy_action["phi"] = noisy_action["phi"] % 360  # 角度循环
            noisy_action["theta"] = np.clip(noisy_action["theta"], 0, 90)
            noisy_action["a"] = np.clip(noisy_action["a"], -0.5, 0.5)
            noisy_action["b"] = np.clip(noisy_action["b"], -0.5, 0.5)

            # 打印原始和噪声后的action（可选）
            logger.info(
                f"Player {self.get_curr_player()} 原始动作: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                f"theta={action['theta']:.2f}°, a={action['a']:.3f}, b={action['b']:.3f}"
            )
            logger.info(
                f"Player {self.get_curr_player()} 实际动作: V0={noisy_action['V0']:.2f}, phi={noisy_action['phi']:.2f}, "
                f"theta={noisy_action['theta']:.2f}°, a={noisy_action['a']:.3f}, b={noisy_action['b']:.3f}"
            )

            # 不启用噪声时，打印原始action
            logger.info(
                f"Player {self.get_curr_player()} 执行指定动作: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                f"theta={action['theta']:.2f}°, a={action['a']:.3f}, b={action['b']:.3f}"
            )

        # 实现击球，通过物理仿真获得击球后的球位置信息
        shot = pt.System(table=self.table, balls=self.balls, cue=self.cue)
        self.cue.set_state(
            V0=action["V0"],
            phi=action["phi"],
            theta=action["theta"],
            a=action["a"],
            b=action["b"],
        )
        pt.simulate(shot, inplace=True)
        # 记录所有shot，用于游戏结束后进行render
        # NOTE: 这里的同样的 System 对象也提供了 copy 方法
        self.shot_record.append(copy.deepcopy(shot))

        # NOTE: 上面的物理模拟结束了，下面开始进行规则判断+状态机更新，我认为有一部分和 agent.py 的 analyze_shot_for_reward 逻辑是重复的

        # 获取 final_states
        # final_states = collect_ball_states(shot)
        # 更新球状态到本次击球后的结果
        # NOTE: 这只是暂时写入。如果后面判定犯规，需要回滚，会再把 self.balls 恢复成 self.last_state
        self.balls = shot.balls
        new_pocketed = [
            bid
            for bid, b in shot.balls.items()
            if b.state.s == 4 and self.last_state[bid].state.s != 4
        ]

        # NOTE: 这一部分也基本一致，没有太大区别，除了 foul_first_hit 这个变量
        events = shot.events
        first_contact_ball_id = None
        # 定义合法的球ID集合（排除 'cue' 和其他非球对象如 'cue stick'）
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

        for e in events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, "ids") else []
            if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
                # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
                other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        cue_hit_cushion = False
        target_hit_cushion = False
        for e in events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, "ids") else []
            if "cushion" in et:
                if "cue" in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True

        # 统计各类结果
        own_pocketed = [
            bid
            for bid in new_pocketed
            if bid in self.player_targets[self.players[self.curr_player]]
        ]
        enemy_pocketed = [
            bid
            for bid in new_pocketed
            if bid not in self.player_targets[self.players[self.curr_player]]
            and bid not in ["cue", "8"]
        ]

        # NOTE: 到这里为止的代码都和 analyze_shot_for_reward 几乎一样，根据事件提取出了我们需要的“事实”（例如new_pocketed，这一杆新进袋了哪些球？；first_contact_ball_id 白球手碰的球是谁？）下面开始有区别了

        ##### 规则判断，是否违规要回退，游戏是否结束，确定下一个击球方  #####

        # 白球和黑8同时落袋即可直接判负
        if "cue" in new_pocketed and "8" in new_pocketed:
            logger.info("⚪+🎱 白球和黑8同时落袋,犯规!判负！")
            logger.info(f"🏆 Player {self.players[1 - self.curr_player]} 获胜！")
            self.done = True
            self.winner = self.players[1 - self.curr_player]
            return {
                "ME_INTO_POCKET": own_pocketed,
                "ENEMY_INTO_POCKET": enemy_pocketed,
                "WHITE_BALL_INTO_POCKET": True,
                "BLACK_BALL_INTO_POCKET": True,
                "FOUL_FIRST_HIT": False,
                "NO_POCKET_NO_RAIL": False,
                "BALLS": copy.deepcopy(self.balls),
            }

        # NOTE: 为什么没有处理“误打黑8”的情况？
        # ! 这里代码并非遗漏，而是先判断了白球进袋的情况，如果白球进袋，那么黑8不可能进袋了（因为黑8和白球同时进袋会在上面的代码里处理），所以可以将误打黑8的情况在下面的代码里处理

        # 白球掉袋 (犯规)
        if "cue" in new_pocketed:
            logger.info("⚪ 白球落袋！犯规，恢复上一杆状态，交换球权。")
            # NOTE: 我觉得这个恢复上一杆状态可以提取成一个方法
            # 保存击打前的balls状态用于返回
            balls_before_shot = copy.deepcopy(self.last_state)
            self.balls = restore_balls_state(self.last_state)
            # NOTE: 交换击球方，这个也应该提取成一个方法
            self.curr_player = 1 - self.curr_player
            self.done = False
            self.hit_count += 1
            if self.hit_count >= self.MAX_HIT_COUNT:
                logger.info("⏰ 达到最大击球数，比赛结束！")
                self.done = True
                a_left = len(
                    [
                        bid
                        for bid in self.player_targets["A"]
                        if bid != "8" and self.balls[bid].state.s != 4
                    ]
                )
                b_left = len(
                    [
                        bid
                        for bid in self.player_targets["B"]
                        if bid != "8" and self.balls[bid].state.s != 4
                    ]
                )
                if a_left < b_left:
                    self.winner = "A"
                elif b_left < a_left:
                    self.winner = "B"
                else:
                    self.winner = "SAME"
                logger.info(
                    f"📊 最大击球数详情：A剩余 {a_left}，B剩余 {b_left}，胜者：{self.winner}"
                )
            return {
                "ME_INTO_POCKET": own_pocketed,
                "ENEMY_INTO_POCKET": enemy_pocketed,
                "WHITE_BALL_INTO_POCKET": True,
                "BLACK_BALL_INTO_POCKET": False,
                "FOUL_FIRST_HIT": False,
                "NO_POCKET_NO_RAIL": False,
                "BALLS": balls_before_shot,
            }

        player = self.get_curr_player()
        remaining_own_before = [
            bid for bid in self.player_targets[player] if self.last_state[bid].state.s != 4
        ]
        # 黑8掉袋 (胜负判断)
        if "8" in new_pocketed:
            # 检查击球前是否已清空所有目标球（不能同时打进最后目标球+黑8）
            if len(remaining_own_before) == 0:
                logger.info(f"🏆 Player {player} 成功打进黑8，获胜！")
                self.winner = self.players[self.curr_player]
            else:
                logger.info(f"💥 Player {player} 误打黑8（自身球未清空），判负！")
                logger.info(f"🏆 Player {self.players[1 - self.curr_player]} 获胜！")
                self.winner = self.players[1 - self.curr_player]
            self.done = True
            return {
                "ME_INTO_POCKET": own_pocketed,
                "ENEMY_INTO_POCKET": enemy_pocketed,
                "WHITE_BALL_INTO_POCKET": False,
                "BLACK_BALL_INTO_POCKET": True,
                "FOUL_FIRST_HIT": False,
                "NO_POCKET_NO_RAIL": False,
                "BALLS": copy.deepcopy(self.balls),
            }

        if first_contact_ball_id is None:
            logger.info("⚠️ 本杆白球未接触任何球，犯规，恢复上一杆状态，交换球权。")
            # 保存击打前的balls状态用于返回
            balls_before_shot = copy.deepcopy(self.last_state)
            self.balls = restore_balls_state(self.last_state)
            self.curr_player = 1 - self.curr_player
            self.hit_count += 1
            if self.hit_count >= self.MAX_HIT_COUNT:
                logger.info("⏰ 达到最大击球数，比赛结束！")
                self.done = True
                a_left = len(
                    [
                        bid
                        for bid in self.player_targets["A"]
                        if bid != "8" and self.balls[bid].state.s != 4
                    ]
                )
                b_left = len(
                    [
                        bid
                        for bid in self.player_targets["B"]
                        if bid != "8" and self.balls[bid].state.s != 4
                    ]
                )
                if a_left < b_left:
                    self.winner = "A"
                elif b_left < a_left:
                    self.winner = "B"
                else:
                    self.winner = "SAME"
                logger.info(
                    f"📊 最大击球数详情：Player A剩余 {a_left}，Player B剩余 {b_left}，胜者：{self.winner}"
                )
            return {
                "ME_INTO_POCKET": own_pocketed,
                "ENEMY_INTO_POCKET": enemy_pocketed,
                "WHITE_BALL_INTO_POCKET": False,
                "BLACK_BALL_INTO_POCKET": False,
                "FOUL_FIRST_HIT": False,
                "NO_POCKET_NO_RAIL": False,
                "NO_HIT": True,
                "BALLS": balls_before_shot,
            }
        if first_contact_ball_id is not None:
            opponent_plus_eight = [
                bid
                for bid in self.balls.keys()
                if bid not in self.player_targets[player] and bid not in ["cue"]
            ]
            if "8" not in opponent_plus_eight:
                opponent_plus_eight.append("8")
            # 当有自己的球剩余时，首次碰撞对方球或黑8犯规
            # 当只剩黑八时，必须首次碰撞黑八，否则碰到对手球也犯规
            if (len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight) or (
                len(remaining_own_before) == 0 and first_contact_ball_id != "8"
            ):
                if len(remaining_own_before) == 0:
                    logger.info(
                        f"⚠️ Player {player} 只剩黑八时首次碰撞非黑八球，犯规，恢复上一杆状态，交换球权。"
                    )
                else:
                    logger.info(
                        f"⚠️ Player {player} 首次碰撞为对方球或黑八，犯规，恢复上一杆状态，交换球权。"
                    )
                # 保存击打前的balls状态用于返回
                balls_before_shot = copy.deepcopy(self.last_state)
                self.balls = restore_balls_state(self.last_state)
                self.curr_player = 1 - self.curr_player
                self.hit_count += 1
                if self.hit_count >= self.MAX_HIT_COUNT:
                    logger.info("⏰ 达到最大击球数，比赛结束！")
                    self.done = True
                    a_left = len(
                        [
                            bid
                            for bid in self.player_targets["A"]
                            if bid != "8" and self.balls[bid].state.s != 4
                        ]
                    )
                    b_left = len(
                        [
                            bid
                            for bid in self.player_targets["B"]
                            if bid != "8" and self.balls[bid].state.s != 4
                        ]
                    )
                    if a_left < b_left:
                        self.winner = "A"
                    elif b_left < a_left:
                        self.winner = "B"
                    else:
                        self.winner = "SAME"
                    logger.info(
                        f"📊 最大击球数详情：A剩余 {a_left}，B剩余 {b_left}，胜者：{self.winner}"
                    )
                return {
                    "ME_INTO_POCKET": own_pocketed,
                    "ENEMY_INTO_POCKET": enemy_pocketed,
                    "WHITE_BALL_INTO_POCKET": False,
                    "BLACK_BALL_INTO_POCKET": False,
                    "FOUL_FIRST_HIT": True,
                    "NO_POCKET_NO_RAIL": False,
                    "BALLS": copy.deepcopy(self.balls),
                }

        # 处理无进球的情况
        if len(new_pocketed) == 0:
            if (not cue_hit_cushion) and (not target_hit_cushion):
                # 无进球且无球碰库，犯规
                logger.info("⚠️ 本杆无进球且母球和目标球均未碰库，犯规，恢复上一杆状态，交换球权。")
                # 保存击打前的balls状态用于返回
                balls_before_shot = copy.deepcopy(self.last_state)
                self.balls = restore_balls_state(self.last_state)
                self.curr_player = 1 - self.curr_player
                self.hit_count += 1
                if self.hit_count >= self.MAX_HIT_COUNT:
                    logger.info("⏰ 达到最大击球数，比赛结束！")
                    self.done = True
                    a_left = len(
                        [
                            bid
                            for bid in self.player_targets["A"]
                            if bid != "8" and self.balls[bid].state.s != 4
                        ]
                    )
                    b_left = len(
                        [
                            bid
                            for bid in self.player_targets["B"]
                            if bid != "8" and self.balls[bid].state.s != 4
                        ]
                    )
                    if a_left < b_left:
                        self.winner = "A"
                    elif b_left < a_left:
                        self.winner = "B"
                    else:
                        self.winner = "SAME"
                    logger.info(
                        f"📊 最大击球数详情：A剩余 {a_left}，B剩余 {b_left}，胜者：{self.winner}"
                    )
                return {
                    "ME_INTO_POCKET": own_pocketed,
                    "ENEMY_INTO_POCKET": enemy_pocketed,
                    "WHITE_BALL_INTO_POCKET": False,
                    "BLACK_BALL_INTO_POCKET": False,
                    "FOUL_FIRST_HIT": False,
                    "NO_POCKET_NO_RAIL": True,
                    "BALLS": balls_before_shot,
                }
            else:
                # 无进球但有球碰库，仅交换球权
                logger.info("⚠️ 本杆无进球，交换球权。")
                self.curr_player = 1 - self.curr_player
                self.last_state = save_balls_state(self.balls)
                self.hit_count += 1
                # NOTE: 这里判断是否超过最大击球数的代码和上面重复了，应该提取成一个方法
                if self.hit_count >= self.MAX_HIT_COUNT:
                    logger.info("⏰ 达到最大击球数，比赛结束！")
                    self.done = True
                    a_left = len(
                        [
                            bid
                            for bid in self.player_targets["A"]
                            if bid != "8" and self.balls[bid].state.s != 4
                        ]
                    )
                    b_left = len(
                        [
                            bid
                            for bid in self.player_targets["B"]
                            if bid != "8" and self.balls[bid].state.s != 4
                        ]
                    )
                    if a_left < b_left:
                        self.winner = "A"
                    elif b_left < a_left:
                        self.winner = "B"
                    else:
                        self.winner = "SAME"
                    logger.info(
                        f"📊 最大击球数详情：A剩余 {a_left}，B剩余 {b_left}，胜者：{self.winner}"
                    )
                return {
                    "ME_INTO_POCKET": own_pocketed,
                    "ENEMY_INTO_POCKET": enemy_pocketed,
                    "WHITE_BALL_INTO_POCKET": False,
                    "BLACK_BALL_INTO_POCKET": False,
                    "FOUL_FIRST_HIT": False,
                    "NO_POCKET_NO_RAIL": False,
                    "BALLS": copy.deepcopy(self.balls),
                }

        # NOTE: 规则 7：有进球时，决定是否继续出杆
        # 判断是否打进自己球，确定下一个击球方
        if own_pocketed:
            logger.info(f"🎯 Player {player} 打进了 {own_pocketed}，继续出杆。")
        else:
            logger.info(f"❌ Player {player} 未打进自己球，交换球权。")
            self.curr_player = 1 - self.curr_player

        # 5. 保存当前状态
        self.last_state = save_balls_state(self.balls)

        # 更新 count数，并且判断数是否过长
        self.hit_count += 1
        if self.hit_count >= self.MAX_HIT_COUNT:
            logger.info("⏰ 达到最大击球数，比赛结束！")
            self.done = True
            a_left = len(
                [
                    bid
                    for bid in self.player_targets["A"]
                    if bid != "8" and self.balls[bid].state.s != 4
                ]
            )
            b_left = len(
                [
                    bid
                    for bid in self.player_targets["B"]
                    if bid != "8" and self.balls[bid].state.s != 4
                ]
            )
            if a_left < b_left:
                self.winner = "A"
            elif b_left < a_left:
                self.winner = "B"
            else:
                self.winner = "SAME"
            logger.info(f"📊 最大击球数详情：A剩余 {a_left}，B剩余 {b_left}，胜者：{self.winner}")
            return {
                "ME_INTO_POCKET": own_pocketed,
                "ENEMY_INTO_POCKET": enemy_pocketed,
                "WHITE_BALL_INTO_POCKET": False,
                "BLACK_BALL_INTO_POCKET": False,
                "BALLS": copy.deepcopy(self.balls),
            }

        # NOTE: 没有任何犯规，也没有进球，没有超过最大击球数
        # return 一些这一杆的结果信息
        return {
            "ME_INTO_POCKET": own_pocketed,
            "ENEMY_INTO_POCKET": enemy_pocketed,
            "WHITE_BALL_INTO_POCKET": False,
            "BLACK_BALL_INTO_POCKET": False,
            "FOUL_FIRST_HIT": False,
            "NO_POCKET_NO_RAIL": False,
            "BALLS": copy.deepcopy(self.balls),
        }
