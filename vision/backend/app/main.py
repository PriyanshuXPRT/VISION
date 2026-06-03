"""FastAPI entrypoint for the V.I.S.I.O.N. hybrid backend."""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import api_router
from backend.app.core import settings
from backend.app.db import init_models


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.name,
        version="0.1.0",
        description=(
            "Hybrid offline-first backend for V.I.S.I.O.N. — handles optional "
            "encrypted sync, device auth, and admin audit access."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.include_router(api_router, prefix="/v1")
    init_models()
    return app


app = create_app()


def main() -> int:
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
