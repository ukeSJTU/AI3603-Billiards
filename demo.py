import pooltool as pt

from src.utils.logger import get_logger, setup_logger

setup_logger(
    file_level="INFO",
    log_to_file=True,
    log_dir="logs",
    log_filename="demo.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format_string="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    diagnose=True,
)

logger = get_logger()

logger.info(f"Pooltool version: {pt.__version__}")
