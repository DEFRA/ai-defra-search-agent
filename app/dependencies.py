import logging

from app import config

logger = logging.getLogger(__name__)


def get_app_config() -> config.AppConfig:
    return config.get_config()
