from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import webvtt
from fastapi import HTTPException

try:
    import yt_dlp
except ImportError:  # pragma: no cover - dependency installed in runtime env
    yt_dlp = None  # type: ignore

from .config import settings


class SubtitleIngestionService:
    """Handle on-demand subtitle downloads and normalisation for single videos."""

    def __init__(self, data_root: Path | None = None, raw_root: Path | None = None):
        self.data_root = data_root or settings.data_root
        self.raw_root = raw_root or settings.raw_root

    def ingest(self, youtube_url: str, domain: Optional[str] = None) -> tuple[str, str, Path]:
        video_id = self._extract_video_id(youtube_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="유효한 YouTube 링크를 입력해주세요.")

        domain_name = domain or settings.default_domain
        raw_dir = self.raw_root / domain_name
        raw_dir.mkdir(parents=True, exist_ok=True)

        caption_path = self._download_subtitles(youtube_url, video_id, raw_dir)
        clean_path = self._convert_to_clean_csv(caption_path, domain_name, video_id)

        return domain_name, video_id, clean_path

    def _download_subtitles(self, youtube_url: str, video_id: str, raw_dir: Path) -> Path:
        if yt_dlp is None:
            raise HTTPException(
                status_code=500,
                detail="yt-dlp 패키지가 설치되어 있지 않습니다. requirements.txt를 설치해주세요.",
            )

        output_template = str(raw_dir / f"{video_id}.%(ext)s")
        ydl_opts = {
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["en"],
            "skip_download": True,
            "outtmpl": output_template,
            "quiet": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
        except Exception as exc:  # pragma: no cover - network failures
            raise HTTPException(status_code=502, detail=f"자막 다운로드 실패: {exc}") from exc

        for suffix in (".en.vtt", ".vtt"):
            candidate = raw_dir / f"{video_id}{suffix}"
            if candidate.exists():
                return candidate

        raise HTTPException(status_code=404, detail="자동 자막을 찾을 수 없습니다.")

    def _convert_to_clean_csv(self, vtt_path: Path, domain: str, video_id: str) -> Path:
        segments = []
        try:
            for caption in webvtt.read(str(vtt_path)):
                text = self._clean_text(caption.text)
                if not text:
                    continue
                segments.append(
                    (
                        caption.start,
                        caption.end,
                        text,
                    )
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VTT 파싱 실패: {exc}") from exc

        if not segments:
            raise HTTPException(status_code=404, detail="유효한 자막 문장을 찾지 못했습니다.")

        output_dir = self.data_root / domain
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_id}.en.csv"

        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["start", "end", "text"])
            writer.writerows(segments)

        return output_path

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = text.replace("\n", " ").strip()
        cleaned = re.sub(r"\[.*?\]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @staticmethod
    def _extract_video_id(youtube_url: str) -> Optional[str]:
        parsed = urlparse(youtube_url)
        if parsed.hostname in {"youtu.be"}:
            return parsed.path.lstrip("/")
        if parsed.hostname and "youtube" in parsed.hostname:
            query = parse_qs(parsed.query)
            if "v" in query:
                return query["v"][0]
            # handle /shorts/<id> etc.
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                return path_parts[-1]
        return None
