from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any
from app.schemas.schemas import UploadSuccess, ErrorResponse
from ..utils.file_validation import save_upload_file, validate_csv_file
from ..utils.model_validation import safe_load_model_from_path
from ..db import get_session
from app.models.models import UploadRecord
from ..config import settings
from pathlib import Path

router = APIRouter(prefix="/api")


@router.post("/upload", response_model=Any)
async def upload_files(
    dataset_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    ds_ext = Path(dataset_file.filename).suffix.lower()
    md_ext = Path(model_file.filename).suffix.lower()

    if ds_ext not in {".csv"}:
        raise HTTPException(status_code=400, detail="Dataset must be a .csv file")
    if md_ext not in {".pkl", ".joblib"}:
        raise HTTPException(
            status_code=400, detail="Model must be a .pkl or .joblib file"
        )

    # save files to temp
    try:
        ds_path = await save_upload_file(dataset_file, subdir="datasets")
        md_path = await save_upload_file(model_file, subdir="models")

        # validate csv
        df, _ = await validate_csv_file(ds_path)

        # validate model
        model_info = safe_load_model_from_path(md_path)

    except ValueError as ve:
        # clean up files if created
        for p in (locals().get("ds_path"), locals().get("md_path")):
            try:
                if p and Path(p).exists():
                    Path(p).unlink()

            except Exception:
                pass

        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    # insert metadata record
    try:
        record = UploadRecord(
            dataset_filename=ds_path.name,
            model_filename=md_path.name,
            dataset_rows=int(df.shape[0]),
            dataset_columns=int(df.shape[1]),
            dataset_columns_list=list(df.columns.astype(str)),
            model_type=model_info["model_type"],
            model_supports_predict_proba=str(model_info["supports_proba"]),
        )

        session.add(record)
        await session.commit()
        await session.refresh(record)

    except Exception:
        record = None

    # response
    success = {
        "status": "success",
        "dataset_info": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns.astype(str)),
        },
        "model_info": {
            "model_type": model_info["model_type"],
            "supports_predic_proba": model_info["supports_proba"],
        },
        "next_step": "select_sensitive_attribute",
    }

    return JSONResponse(content=success)
