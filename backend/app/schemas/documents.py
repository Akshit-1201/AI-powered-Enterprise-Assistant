"""Schemas for document upload/listing."""
import datetime as dt

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    chunk_count: int
    uploaded_at: dt.datetime


class UploadResponse(BaseModel):
    id: int
    filename: str
    chunk_count: int
    message: str = "Document uploaded and indexed."
