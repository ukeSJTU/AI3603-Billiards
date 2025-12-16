"""
logger.py - Loguru 配置模块

功能：
- 控制台输出 INFO 级别（彩色）
- 文件输出 DEBUG 级别
- 日志文件名包含运行时间
- 日志目录: logs/
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# 标记是否已初始化
_initialized = False


def setup_logger(
    log_dir: str = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> Path:
    """
    配置 loguru logger

    Args:
        log_dir: 日志目录路径
        console_level: 控制台日志级别
        file_level: 文件日志级别

    Returns:
        Path: 日志文件路径
    """
    global _initialized

    if _initialized:
        return Path(log_dir)

    # 移除默认 handler
    logger.remove()

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 生成日志文件名（包含运行时间）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"train_{timestamp}.log"

    # 控制台 handler: INFO 级别，彩色输出
    logger.add(
        sys.stderr,
        level=console_level,
        format="<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        colorize=True,
    )

    # 文件 handler: DEBUG 级别，完整信息
    logger.add(
        str(log_file),
        level=file_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}",
        rotation="100 MB",  # 日志文件超过 100MB 时轮转
        retention="7 days",  # 保留 7 天的日志
        compression="zip",  # 压缩旧日志
        encoding="utf-8",
    )

    _initialized = True
    logger.info(f"日志系统已初始化，文件输出: {log_file}")

    return log_file


def get_logger(name: str | None = None):
    """
    获取带有特定名称的 logger

    Args:
        name: logger 名称（通常是模块名 __name__）

    Returns:
        配置好的 logger 实例
    """
    if name:
        return logger.bind(name=name)
    return logger


# 模块级便捷导出
__all__ = ["setup_logger", "get_logger", "logger"]
