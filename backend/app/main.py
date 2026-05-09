from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .db import engine
from . import models
from .routers.upload import router as upload_router
from .routers.bias import router as bias_router
from .routers.bias_mitigation import router as bias_mitigation_router
from .routers.correction import router as correction_router
from .routers.optimization import router as optimization_router
from .routers.experiments import router as experiments_router
from .routers.explainability import router as explainability_router
from .routers.retraining import router as retraining_router
from .routers.model_registry import router as model_registry_router


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
app.include_router(bias_mitigation_router)
app.include_router(correction_router)
app.include_router(optimization_router)
app.include_router(experiments_router)
app.include_router(explainability_router)
app.include_router(retraining_router)
app.include_router(model_registry_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
