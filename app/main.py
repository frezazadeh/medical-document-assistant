import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import protected, public
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.llm import LLMError
from app.observability.logging import configure_logging, request_id_var

logger = logging.getLogger("app.api")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Built once at startup: loading the embedding model takes a few seconds.
        app.state.container = container or build_container(settings)
        logger.info(
            "service started",
            extra={
                "llm_provider": app.state.container.llm.provider,
                "llm_model": app.state.container.llm.model,
            },
        )
        yield

    app = FastAPI(
        title="Medical Document Assistant",
        description="Upload clinical documents and ask questions with cited answers.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return response

    @app.exception_handler(LLMError)
    async def llm_error(request: Request, exc: LLMError):
        logger.error("llm backend failed", extra={"error": str(exc)})
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    app.include_router(public)
    app.include_router(protected)
    return app


app = create_app()
