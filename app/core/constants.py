from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class SourceKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    ZIP = "zip"
