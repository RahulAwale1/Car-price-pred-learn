from pydantic import BaseModel
class ErrorDetail(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None

class ErrorResponse(BaseModel):
    detail: ErrorDetail