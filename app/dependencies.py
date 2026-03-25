import logging

import fastapi
import fastapi.security
from fastapi import Security, status

from app import config

logger = logging.getLogger(__name__)

_api_key_header = fastapi.security.APIKeyHeader(name="X-API-KEY", auto_error=False)


def get_app_config() -> config.AppConfig:
    return config.get_config()


def verify_api_key(key: str | None = Security(_api_key_header)) -> None:
    app_config = config.get_config()
    if key is None:
        raise fastapi.HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if key != app_config.inbound_api_key:
        raise fastapi.HTTPException(status_code=status.HTTP_403_FORBIDDEN)
