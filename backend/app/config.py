from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "BiasBuster"
    DEBUG: bool = True

    database_url: str  # ← lowercase field

    TEMP_DIR: str = "/tmp/biasbuster_uploads"
    MAX_CSV_SIZE_BYTES: int = 50 * 1024 * 1024

    model_config = {"env_file": Path.cwd() / ".env"}

    @property
    def DATABASE_URL(self):
        return self.database_url  # allows settings.DATABASE_URL to work


settings = Settings()
