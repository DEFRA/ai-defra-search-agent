import logging
from functools import lru_cache

from app import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_app_config() -> config.AppConfig:
    return config.get_config()
