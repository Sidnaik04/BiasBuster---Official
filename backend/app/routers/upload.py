from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any
from pathlib import Path
from ..utils.file_validation import save_upload_file, validate_csv_file
from ..utils.model_validation import safe_load_model_from_path
from ..db import get_session
from app.models.models import UploadRecord

router = APIRouter(prefix="/api")

@router.post("/upload", response_model=Any)
async def upload_files(
    dataset_file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    ds_ext = Path(dataset_file.filename).suffix.lower()
    md_ext = Path(model_file.filename).suffix.lower()

    if ds_ext != ".csv":
        raise HTTPException(status_code=400, detail="Dataset must be a .csv file")
    if md_ext not in {".pkl", ".joblib"}:
        raise HTTPException(status_code=400, detail="Model must be a .pkl or .joblib file")

    try:
        ds_path = await save_upload_file(dataset_file, subdir="datasets")
        md_path = await save_upload_file(model_file, subdir="models")

        df, _ = await validate_csv_file(ds_path)
        model_info = safe_load_model_from_path(md_path)

    except ValueError as ve:
        for p in (locals().get("ds_path"), locals().get("md_path")):
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except:
                pass
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    # FIXED HERE ↓ store Boolean, no str() wrapping
    record = UploadRecord(
        dataset_filename = ds_path.name,
        model_filename = md_path.name,
        dataset_rows = int(df.shape[0]),
        dataset_columns = int(df.shape[1]),
        dataset_columns_list = df.columns.astype(str).tolist(),
        model_type = model_info["model_type"],
        model_supports_predict_proba = bool(model_info["supports_proba"]),  # ✔ Boolean
    )

    session.add(record)
    await session.commit()
    await session.refresh(record)

    success = {
        "status": "success",
        "dataset_info": {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
        },
        "model_info": {
            "model_type": model_info["model_type"],
            "supports_predict_proba": model_info["supports_proba"],  # ✔ fixed typo
        },
        "next_step": "select_sensitive_attribute",
    }

    return JSONResponse(content=success)
