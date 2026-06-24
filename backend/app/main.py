"""FastAPI application entry point.

Run from the backend/ directory:
    uvicorn app.main:app --reload
"""
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import ask, auth, documents, health
from app.config import get_settings
from app.db.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


def create_app() -> FastAPI:
    settings = get_settings()
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    init_db()

    app = FastAPI(title="Enterprise AI Assistant", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # Safety net: never leak a stack trace to the client. Specific cases (missing key,
        # provider outage, validation) are handled closer to where they occur.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(ask.router)
    app.include_router(documents.router)
    return app


app = create_app()
