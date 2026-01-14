from pydantic import BaseModel


class DomainSummary(BaseModel):
    domain: str
    video_count: int


class VideoSummary(BaseModel):
    domain: str
    video_id: str
    caption_path: str
    segment_count: int


class Segment(BaseModel):
    segment_id: int
    start: str
    end: str
    start_sec: float
    end_sec: float
    text: str


class VideoSegmentsResponse(BaseModel):
    video: VideoSummary
    segments: list[Segment]


class QARequest(BaseModel):
    question: str
    domain: str | None = None
    video_id: str | None = None


class QAResponse(BaseModel):
    message: str
    detail: str | None = None
