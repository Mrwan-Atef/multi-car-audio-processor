from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router, stt_service
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    stt_service.load()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(router)


@app.get('/')
def root() -> dict:
    return {
        'service': settings.app_name,
        'version': settings.app_version,
        'status': 'running',
    }
