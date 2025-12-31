# RandomAgent 就是一个纯随机的 Agent，只是测试用的

# TODO：也许可以把 RandomAgent 和 BasicAgent 对打，说明至少简单决策也比纯随机好？

from typing import Any, List, Optional, override

from .base import ActionDict, Agent, BallsDict


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    @override
    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[Any] = None,
    ) -> ActionDict:
        return self._random_action()
