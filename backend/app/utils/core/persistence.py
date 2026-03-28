def get_latest_model(upload_id, db, base_model_path):
    """
    Returns the most recent mitigated model if available,
    otherwise returns the original uploaded model.
    """

    from app.models.bias_mitigation import BiasMitigationRun

    latest = (
        db.query(BiasMitigationRun)
        .filter(BiasMitigationRun.upload_id == upload_id)
        .order_by(BiasMitigationRun.created_at.desc())
        .first()
    )

    if latest:
        return latest.artifact_model_path

    return base_model_path
