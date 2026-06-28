from fastapi import HTTPException


def raise_api_error(
        status_code: int,
        error_code: str,
        message: str,
        request_id: str | None = None
):
    raise HTTPException(
        status_code=status_code,
        detail={
            "error_code": error_code,
            "message": message,
            "request_id": request_id
        }
    )
