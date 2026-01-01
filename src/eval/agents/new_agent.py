# ruff: noqa
# fmt: off

"""
NewAgent - 用于评测的新Agent包装器
直接使用 GeometryAgent
"""

from .geometry_agent import GeometryAgent


class NewAgent(GeometryAgent):
    """
    NewAgent 继承自 GeometryAgent
    配置参数根据 configs/basic_vs_geometry.yaml
    """
    def __init__(self):
        # 使用配置文件中的参数
        super().__init__(
            n_candidates=30,
            angle_spread=5.0,
            v0_spread=1.0
        )
