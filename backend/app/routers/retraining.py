from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.retraining.retraining_pipeline import run_retraining
from app.config import settings
from app.db import get_session
from app.models.models import UploadRecord
from app.utils.model_validation import safe_load_model_from_path
import pandas as pd

router = APIRouter()


class RetrainRequest(BaseModel):
    upload_id: int
    target_column: str
    sensitive_columns: List[str]
    strategy: str
    train_additional_models: bool = True


@router.post("/api/retraining/run")
async def run(request: RetrainRequest, session: AsyncSession = Depends(get_session)):
    # Resolve upload filenames from DB record, then locate file in TEMP_DIR first.
    try:
        record = (
            await session.execute(
                select(UploadRecord).where(UploadRecord.id == request.upload_id)
            )
        ).scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Upload record not found")

        temp_uploads_dir = Path(settings.TEMP_DIR)
        artifact_dir = Path(settings.ARTIFACT_DIR)

        dataset_path = temp_uploads_dir / "datasets" / record.dataset_filename
        if not dataset_path.exists():
            dataset_path = artifact_dir / "datasets" / record.dataset_filename

        model_path = temp_uploads_dir / "models" / record.model_filename
        if not model_path.exists():
            model_path = artifact_dir / "models" / record.model_filename

        if not dataset_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"dataset='{dataset_path}' model='{model_path}'")

        dataset = pd.read_csv(dataset_path)
        uploaded_model = safe_load_model_from_path(model_path)["model"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Upload files not found: {e}")

    if request.strategy not in ("reweighting", "smote"):
        raise HTTPException(
            status_code=400, detail="Unsupported strategy for retraining"
        )

    result = run_retraining(
        upload_dataset=dataset,
        uploaded_model=uploaded_model,
        target_column=request.target_column,
        sensitive_columns=request.sensitive_columns,
        strategy=request.strategy,
        train_additional_models=request.train_additional_models,
    )

    return result.__dict__
