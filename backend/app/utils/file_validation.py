import os
import aiofiles
import uuid
from pathlib import Path
from typing import Tuple
import pandas as pd
from pandas.errors import EmptyDataError

from ..config import settings

TEMP_DIR = Path(settings.TEMP_DIR)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_DATA_EXT = {".csv"}
ALLO_MODEL_EXT = {".pkl", ".joblib"}


async def save_upload_file(upload_file, subdir: str = "") -> Path:
    file_ext = Path(upload_file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{file_ext}"

    dir_path = TEMP_DIR / subdir
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / unique_name

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await upload_file.read()
        await out_file.write(content)

    await upload_file.seek(0)
    return file_path


async def validate_csv_file(file_path: Path) -> Tuple[pd.DataFrame, str]:
    # read using pandas from path (sync)
    try:
        df = pd.read_csv(file_path)

    except UnicodeDecodeError as e:
        raise ValueError(
            "Unreadable CSV encoding. Please upload UTF-8 encoded CSV."
        ) from e

    except EmptyDataError as e:
        raise ValueError("CSV is empty.") from e

    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}") from e

    if df.empty:
        raise ValueError("CSV contains no rows")

    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least 2 columns (features + target)")

    if file_path.stat().st_size > settings.MAX_CSV_SIZE_BYTES:
        raise ValueError("CSV file exceeds maximum allowed size")

    return df, "ok"
