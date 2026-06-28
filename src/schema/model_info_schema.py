from pydantic import BaseModel, Field, ConfigDict
class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    input_schema_version: str
    currency: str
    model_loaded: bool