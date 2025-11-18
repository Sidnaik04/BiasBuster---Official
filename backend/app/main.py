from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .routers.upload import router as upload_router
from .db import engine
from . import models

app = FastAPI(title="BiasBuster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    yield


# routers
app.include_router(upload_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
