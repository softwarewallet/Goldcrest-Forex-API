import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    key = settings.api_key.get_secret_value()
    if not settings.api_key_configured:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
