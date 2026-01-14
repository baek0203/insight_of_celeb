from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .data_access import DataRepository
from .models import (
    DomainSummary,
    QARequest,
    QAResponse,
    VideoSegmentsResponse,
    VideoSummary,
)
from .services import SubtitleIngestionService


app = FastAPI(
    title="Subtitle Intelligence API",
    version="0.1.0",
    description="Endpoints for exploring cleaned subtitles prior to QA backend integration.",
)

repository = DataRepository(settings.data_root)
ingestion_service = SubtitleIngestionService(settings.data_root, settings.raw_root)

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
static_dir = base_dir / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def homepage(
    request: Request,
    youtube_url: str | None = None,
    domain: str | None = None,
    video_id: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    context = {
        "request": request,
        "youtube_url": youtube_url,
        "domain": domain,
        "video_id": video_id,
        "error": error,
    }
    return templates.TemplateResponse("index.html", context)


@app.post("/", response_class=HTMLResponse)
def ingest_video(
    request: Request,
    youtube_url: str = Form(...),
    domain: str | None = Form(None),
) -> HTMLResponse:
    context = {
        "request": request,
        "youtube_url": youtube_url,
        "domain": domain,
        "video_id": None,
        "error": None,
    }

    try:
        domain_name, video_id, _ = ingestion_service.ingest(youtube_url, domain)
        video_data = repository.load_video(domain_name, video_id)
        context.update(
            {
                "domain": domain_name,
                "video_id": video_id,
                "segment_count": video_data.video.segment_count,
            }
        )
    except HTTPException as exc:
        context["error"] = exc.detail

    return templates.TemplateResponse("index.html", context)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/domains", response_model=list[DomainSummary])
def list_domains() -> list[DomainSummary]:
    return repository.list_domains()


@app.get("/domains/{domain}/videos", response_model=list[VideoSummary])
def list_videos(domain: str) -> list[VideoSummary]:
    try:
        return repository.list_videos(domain)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/domains/{domain}/videos/{video_id}/segments",
    response_model=VideoSegmentsResponse,
)
def get_video_segments(domain: str, video_id: str) -> VideoSegmentsResponse:
    try:
        return repository.load_video(domain, video_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/ask", response_model=QAResponse, status_code=202)
def ask_question(payload: QARequest) -> QAResponse:
    return QAResponse(
        message="Question answering pipeline is not connected yet.",
        detail="Endpoint reserved for upcoming retriever+LLM integration.",
    )
