from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .db import engine
from . import models
from .routers.upload import router as upload_router
from .routers.bias import router as bias_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    yield


app = FastAPI(title="BiasBuster API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# routers
app.include_router(upload_router)
app.include_router(bias_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
