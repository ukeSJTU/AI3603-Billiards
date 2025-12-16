"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import copy
import random
import signal

import numpy as np
import pooltool as pt

# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""

    pass


def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")


# NOTE: this protection exists to prevent the pooltool simulation from hanging indefinitely and blocking the program.
def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟

    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒

    返回：
        bool: True 表示模拟成功，False 表示超时或失败

    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    # ? I suspect this will not work on Windows systems, as signal.SIGALRM is not supported there. And it also generally works reliably only in the main thread.
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间

    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器


# ============================================


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
    # NOTE: new_pocketed 包含所有新进袋的球ID(当前状态为 4-pocketed，且上次状态非 pocketed)
    # BallState.s: The motion state label of the ball.
    # possible values can be found here: https://pooltool.readthedocs.io/en/latest/autoapi/pooltool/objects/index.html#pooltool.objects.BallState.s
    new_pocketed = [
        bid
        for bid, b in shot.balls.items()
        if b.state.s == 4 and last_state[bid].state.s != 4
    ]

    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    # NOTE: own_pocketed 指的是当前这一轮击球中，己方进的球
    # ? 为什么计算 own_pockted 的时候不需要排除 "cue" 和 "8" 呢？
    # NOTE: 因为 player_targets 本身就不会包含 "cue" 和 "8"（除非清台后只剩 "8"）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    # NOTE: enemy_pocketed 指的是当前这一轮击球中，对方进的球，不包括白球和黑8
    enemy_pocketed = [
        bid
        for bid in new_pocketed
        if bid not in player_targets and bid not in ["cue", "8"]
    ]

    # NOTE: 白球和黑8进袋单独处理，方便后续计算
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

    # NOTE: 下面这个循环是想要找到第一个母球碰撞的目标球ID，记录在 first_contact_ball_id 中，这个不一定是 player_targets 里的球，后续会进行判定
    # NOTE: shot.events 是按照时间顺序排列的事件列表，包括碰撞、进袋等事件
    for e in shot.events:
        # NOTE: possible values for event_type can be found here: https://pooltool.readthedocs.io/en/latest/autoapi/pooltool/events/index.html#pooltool.events.EventType
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if ("cushion" not in et) and ("pocket" not in et) and ("cue" in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != "cue" and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                # NOTE: 只关心首次碰撞，找到后立即跳出循环
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
    # NOTE: cue_hit_cushion 记录是否母球碰库
    cue_hit_cushion = False
    # NOTE: target_hit_cushion 记录第一次触碰（first_contact_ball_id）的球是否碰库
    target_hit_cushion = False
    foul_no_rail = False

    # NOTE: 再次扫描事件列表，检查有没有和库相关的事件，并更新 cue_hit_cushion 和 target_hit_cushion
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, "ids") else []
        if "cushion" in et:
            if "cue" in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    # NOTE: 碰库犯规判定，当以下条件同时满足时，判定为未碰库犯规：
    # - 本次击球没有任何球进袋（len(new_pocketed) == 0）
    # - 首次碰撞的球存在（first_contact_ball_id is not None）
    # - 母球没有碰库（not cue_hit_cushion）
    # - 首次碰撞的球没有碰库（not target_hit_cushion）
    if (
        len(new_pocketed) == 0
        and first_contact_ball_id is not None
        and (not cue_hit_cushion)
        and (not target_hit_cushion)
    ):
        foul_no_rail = True

    # NOTE: 上面是按照规则分析，下面开始计算得分

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


class Agent:
    """Agent 基类"""

    def __init__(self):
        pass

    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）

        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass

    # NOTE: 这个可以看作“兜底动作生成器”，当Agent算不出来、或者初始化阶段需要随机探索时会调用它，同时也定义了整个项目的连续动作空间，所有 Agents 的 decision 方法都应该返回符合这个范围的动作
    def _random_action(
        self,
    ):
        """生成随机击球动作

        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        # NOTE: 这里的round我的理解是对于连续动作空间做了一个离散化处理，方便贝叶斯优化器搜索，打印日志时也更美观
        action = {
            "V0": round(random.uniform(0.5, 8.0), 2),  # 初速度 0.5~8.0 m/s
            "phi": round(random.uniform(0, 360), 2),  # 水平角度 (0°~360°)
            "theta": round(random.uniform(0, 90), 2),  # 垂直角度
            "a": round(
                random.uniform(-0.5, 0.5), 3
            ),  # 杆头横向偏移（单位：球半径比例）
            "b": round(random.uniform(-0.5, 0.5), 3),  # 杆头纵向偏移
        }
        return action


class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""

    def __init__(self, target_balls=None):
        """初始化 Agent

        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()

        # NOTE: 这个是贝叶斯优化的参数搜索范围，必须和环境允许的动作范围一致
        # ? 我感觉这个应该提取到统一的 config 文件里面去
        # 搜索空间
        self.pbounds = {
            "V0": (0.5, 8.0),
            "phi": (0, 360),
            "theta": (0, 90),
            "a": (-0.5, 0.5),
            "b": (-0.5, 0.5),
        }

        # NOTE: INITIAL_SEARCH 表示初始随机采样点数量，OPT_SEARCH 表示后续优化迭代次数，ALPHA 是高斯过程回归的噪声参数。
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2

        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {"V0": 0.1, "phi": 0.1, "theta": 0.1, "a": 0.003, "b": 0.003}
        self.enable_noise = False

        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        NOTE: 创建一个 `BayesianOptimization` 对象，让它在 pbounds 这个 5 维连续空间里反复调用 `reward_function` 评估动作好坏，并用高斯过程（GP）建模“参数→得分”的函数，从而更聪明地选下一次试探的参数。

        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子

        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed,
        )

        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8, gamma_pan=1.0
        )

        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer,
        )
        # ? 没有不用私有属性的方法吗？
        optimizer._gp = gpr

        return optimizer

    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数

        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象

        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        # NOTE: Step 0: 检查输入，如果没有 balls 信息则直接返回随机动作
        if balls is None:
            print("[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            # NOTE: Step 1: 保存击球前快照 + 判断是否该打 8
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {
                bid: copy.deepcopy(ball) for bid, ball in balls.items()
            }

            # ? 我感觉这里这个提示：Object of type "None" cannot be used as iterable value 是因为 my_targets 可能是 None 导致的，那么我们要不要在 step 0 就检查 my_targets 是否为 None 呢？一起处理？
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            # NOTE: 如果己方目标球已全部进袋，则切换目标为 8 号球
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # NOTE: Step 2: 定义奖励函数并运行贝叶斯优化器搜索最佳参数
            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            # NOTE: 这个 reward_fn_wrapper 函数的作用是：对于给定的一组击球参数 (V0, phi, theta, a, b)，在沙盒环境中模拟击球过程，并使用 analyze_shot_for_reward 函数对结果进行评分，返回该评分作为奖励值。
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                # NOTE: 这里利用 copy.deepcopy 深拷贝球和桌子对象，确保每次模拟互不干扰
                # ? 为什么不用 pt 对象自带的 copy 方法？例如：https://pooltool.readthedocs.io/en/latest/autoapi/pooltool/objects/index.html#pooltool.objects.Table.copy 。从库代码注释来看，库实现不是完全的深拷贝，而是考虑到性能做出优化的拷贝（只深拷贝可变对象，共享不可变对象）。类似的有 Ball 对象的 copy 方法：https://pooltool.readthedocs.io/en/latest/autoapi/pooltool/objects/index.html#pooltool.objects.Ball.copy
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std["V0"])
                        phi_noisy = phi + np.random.normal(0, self.noise_std["phi"])
                        theta_noisy = theta + np.random.normal(
                            0, self.noise_std["theta"]
                        )
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
                        # NOTE: 我觉得这个和上面的重复了，可以优化
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)

                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception:
                    # ? 我看 set_state 应该是不会跑出异常的，那么这里应该只能捕获 simulate_with_timeout 里面的异常了吧？如果是这样的话，为什么不直接在 simulate_with_timeout 里面处理异常然后返回 False 呢？
                    # 模拟失败，给予极大惩罚
                    return -500

                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot, last_state=last_state_snapshot, player_targets=my_targets
                )

                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")

            # NOTE: Step 3：创建优化器并跑 maximize
            # ? 这个应该是 np.random.randint(int(1e6))，生成一个 0~999999 之间的整数作为随机种子
            seed = np.random.randint(1e6)  # type: ignore
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            # NOTE: 先随机采样 INITIAL_SEARCH 次，然后基于采样结果进行 OPT_SEARCH 次优化迭代
            optimizer.maximize(init_points=self.INITIAL_SEARCH, n_iter=self.OPT_SEARCH)

            # NOTE: Step 4：拿到当前找到的最优参数
            best_result = optimizer.max
            # ? 我这里提示 Object of type "None" is not subscriptable，是因为 optimizer.max 可能是 None 吗？后续是否要处理，比如回退到随机动作？
            best_params = best_result["params"]
            best_score = best_result["target"]

            # NOTE: Step 5：低分兜底、否则返回最优动作
            # NOTE: 如果连“合法无进球的小奖励 10”都达不到，认为没搜到靠谱方案，干脆随机。
            if best_score < 10:
                print(
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

            print(
                f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}"
            )
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback

            traceback.print_exc()
            return self._random_action()


class NewAgent(Agent):
    """自定义 Agent 模板（待学生实现）"""

    def __init__(self):
        pass

    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法

        参数：
            observation: (balls, my_targets, table)

        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        return self._random_action()
