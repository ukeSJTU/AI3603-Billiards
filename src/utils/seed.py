import random

import numpy as np

from src.utils.logger import get_logger

try:
    import torch
except ImportError:
    torch = None


logger = get_logger()


def set_random_seed(enable=False, seed=42):
    """设置随机种子以确保实验的可重复性

    Args:
        enable (bool, optional): 是否启用固定随机种子. Defaults to False.
        seed (int, optional): 当 enable 为 True 时使用的随机种子. Defaults to 42.
    """
    if enable:
        # 设置 Python 随机种子
        random.seed(seed)
        # 设置 NumPy 随机种子
        np.random.seed(seed)

        # 设置 PyTorch 随机种子（如果可用）
        if torch is not None:
            torch.manual_seed(seed)  # CPU 随机种子

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)  # 当前 GPU 随机种子
                torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子
                # 确保 CUDA 操作的确定性
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        logger.info(f"随机种子已设置为: {seed}")
    else:
        # 重置为随机性，使用系统时间作为种子
        random.seed()
        np.random.seed(None)

        logger.info("随机种子已禁用，使用完全随机模式")
