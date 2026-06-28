from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List

class CarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symboling: int = Field(...)
    wheelbase: float = Field(...)
    carlength: float = Field(...)
    carwidth: float = Field(...)
    carheight: float = Field(...)
    curbweight: int = Field(...)
    enginesize: int = Field(...)
    boreratio: float = Field(...)
    stroke: float = Field(...)
    compressionratio: float = Field(...)
    horsepower: int = Field(...)
    peakrpm: int = Field(..., ge=0)
    citympg: int = Field(..., ge=0)
    highwaympg: int = Field(...)

    # Categorical features
    fueltype: Literal["gas", "diesel"] = Field(...)
    aspiration: str = Field(...)
    doornumber: str = Field(...)
    carbody: str = Field(...)
    drivewheel: str = Field(...)
    enginelocation: str = Field(...)
    enginetype: str = Field(...)
    cylindernumber: str = Field(...)
    fuelsystem: str = Field(...)
    brand: str = Field(...)