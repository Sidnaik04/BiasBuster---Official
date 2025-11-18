try:
    from pydantic_settings import BaseSettings
except Exception:
    from pydantic import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "BiasBuster"
    DEBUG: bool = True
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/biasbuster_db"
    )
    TEMP_DIR: str = "/tmp/biasbuster_uploads"
    MAX_CSV_SIZE_BYTES: int = 50 * 1024 * 1024

    model_config = {"env_file": Path.cwd() / ".env"}


settings = Settings()
