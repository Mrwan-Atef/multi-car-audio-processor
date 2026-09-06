from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Multi-Car Audio AI Service",
    version="1.0.0",
)


app.include_router(router)


@app.get("/")
def root():

    return {
        "service": (
            "multi-car-audio-processor"
        ),
        "status": "running",
    }