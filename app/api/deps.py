import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.container import Container

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_api_key(
    api_key: str | None = Depends(_api_key_header),
    container: Container = Depends(get_container),
) -> None:
    """A single shared key is enough to show where auth sits in a prototype.

    With real users this would be OAuth2/JWT with per-user scopes, so that
    document access can be tied to an identity and audited.
    """
    expected = container.settings.api_key
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
