from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pathlib import Path
import os

from app.db import get_session
from app.models.models import CorrectionRecord as CorrectionRecordModel, UploadRecord
from app.schemas.correction import (
    DataCorrectionWizardRequest,
    DataCorrectionWizardResponse,
    CorrectionRecord,
)
from app.services.correction_service import run_data_correction_wizard
from app.config import settings

router = APIRouter(prefix="/api/correction", tags=["Data Correction"])


@router.post("/run", response_model=DataCorrectionWizardResponse)
async def run_correction_wizard(
    payload: DataCorrectionWizardRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Run the Data Correction Wizard.

    Applies selected mitigation strategy, generates corrected dataset and retrained model,
    and returns before/after metrics with export paths.

    Args:
        payload: Request with upload_id, target_column, sensitive_columns, strategy

    Returns:
        DataCorrectionWizardResponse with correction results and artifact paths

    Strategies:
        - threshold: Adjust prediction thresholds per group
        - reweighting: Reweight samples inversely proportional to group size
        - smote: Oversample underrepresented groups (requires imbalanced-learn)
    """
    try:
        # Validate upload record exists
        record = (
            await session.execute(
                select(UploadRecord).where(UploadRecord.id == payload.upload_id)
            )
        ).scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Upload record not found")

        # Validate strategy
        if payload.strategy not in ["threshold", "reweighting", "smote"]:
            raise HTTPException(
                status_code=400,
                detail="Strategy must be 'threshold', 'reweighting', or 'smote'",
            )

        # Run wizard
        result = await run_data_correction_wizard(payload, session)

        # Store correction record in database
        correction_record = CorrectionRecordModel(
            correction_id=result["correction_id"],
            upload_id=payload.upload_id,
            strategy=payload.strategy,
            target_column=payload.target_column,
            sensitive_columns=payload.sensitive_columns,
            dataset_export_path=result["dataset_export_path"],
            model_export_path=result["model_export_path"],
            report_export_path=result["report_export_path"],
            metrics_before=result["metrics_before"],
            metrics_after=result["metrics_after"],
            improvements=result["improvements"],
            summary=result["summary"],
            status="success",
        )
        session.add(correction_record)
        await session.commit()

        return DataCorrectionWizardResponse(**result)

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data correction failed: {str(e)}")


@router.get("/download-dataset/{correction_id}")
async def download_corrected_dataset(
    correction_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Download corrected dataset as CSV.

    Args:
        correction_id: Unique correction identifier

    Returns:
        CSV file download
    """
    try:
        # Find correction record
        record = (
            await session.execute(
                select(CorrectionRecordModel).where(
                    CorrectionRecordModel.correction_id == correction_id
                )
            )
        ).scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Correction record not found")

        if not record.dataset_export_path:
            raise HTTPException(
                status_code=404,
                detail="Corrected dataset not available for this strategy",
            )

        dataset_path = Path(record.dataset_export_path)
        if not dataset_path.exists():
            raise HTTPException(
                status_code=404, detail="Corrected dataset file not found on server"
            )

        return FileResponse(
            path=dataset_path,
            media_type="text/csv",
            filename=f"{correction_id}_dataset.csv",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-model/{correction_id}")
async def download_corrected_model(
    correction_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Download corrected/retrained model as joblib file.

    Args:
        correction_id: Unique correction identifier

    Returns:
        .joblib file download
    """
    try:
        # Find correction record
        record = (
            await session.execute(
                select(CorrectionRecordModel).where(
                    CorrectionRecordModel.correction_id == correction_id
                )
            )
        ).scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Correction record not found")

        if not record.model_export_path:
            raise HTTPException(status_code=404, detail="Corrected model not available")

        model_path = Path(record.model_export_path)
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail="Corrected model file not found on server"
            )

        return FileResponse(
            path=model_path,
            media_type="application/octet-stream",
            filename=f"{correction_id}_model.joblib",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-report/{correction_id}")
async def download_correction_report(
    correction_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Download correction report as JSON metadata.

    Args:
        correction_id: Unique correction identifier

    Returns:
        JSON file download
    """
    try:
        # Find correction record
        record = (
            await session.execute(
                select(CorrectionRecordModel).where(
                    CorrectionRecordModel.correction_id == correction_id
                )
            )
        ).scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Correction record not found")

        if not record.report_export_path:
            raise HTTPException(
                status_code=404, detail="Correction report not available"
            )

        report_path = Path(record.report_export_path)
        if not report_path.exists():
            raise HTTPException(
                status_code=404, detail="Correction report file not found on server"
            )

        return FileResponse(
            path=report_path,
            media_type="application/json",
            filename=f"{correction_id}_report.json",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{correction_id}", response_model=CorrectionRecord)
async def get_correction_record(
    correction_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieve a specific correction record with all metadata.

    Args:
        correction_id: Unique correction identifier

    Returns:
        CorrectionRecord with full details
    """
    try:
        record = (
            await session.execute(
                select(CorrectionRecordModel).where(
                    CorrectionRecordModel.correction_id == correction_id
                )
            )
        ).scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Correction record not found")

        return CorrectionRecord(
            id=record.id,
            correction_id=record.correction_id,
            upload_id=record.upload_id,
            strategy=record.strategy,
            target_column=record.target_column,
            sensitive_columns=record.sensitive_columns,
            dataset_export_path=record.dataset_export_path,
            model_export_path=record.model_export_path,
            report_export_path=record.report_export_path,
            metrics_before=record.metrics_before,
            metrics_after=record.metrics_after,
            improvements=record.improvements,
            summary=record.summary,
            status=record.status,
            error_message=record.error_message,
            created_at=record.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_correction_history(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieve correction history for a specific upload.

    Args:
        upload_id: UploadRecord ID

    Returns:
        List of CorrectionRecords
    """
    try:
        records = (
            (
                await session.execute(
                    select(CorrectionRecordModel)
                    .where(CorrectionRecordModel.upload_id == upload_id)
                    .order_by(CorrectionRecordModel.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

        return {
            "upload_id": upload_id,
            "corrections": [
                {
                    "correction_id": r.correction_id,
                    "strategy": r.strategy,
                    "status": r.status,
                    "fairness_gain": r.improvements.get("fairness_gain", 0),
                    "accuracy_change": r.improvements.get("accuracy_change", 0),
                    "summary": r.summary,
                    "created_at": r.created_at,
                }
                for r in records
            ],
            "total": len(records),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
