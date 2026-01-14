from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from .models import DomainSummary, Segment, VideoSegmentsResponse, VideoSummary


class DataRepository:
    """Lightweight reader for per-video subtitle artefacts."""

    def __init__(self, data_root: Path):
        self.data_root = data_root

    def list_domains(self) -> List[DomainSummary]:
        domains = []
        for path in sorted(self.data_root.iterdir()):
            if path.is_dir():
                video_ids = {
                    self._extract_video_id(file_path.name)
                    for file_path in path.glob("*.csv")
                }
                domains.append(DomainSummary(domain=path.name, video_count=len(video_ids)))
        return domains

    def list_videos(self, domain: str) -> List[VideoSummary]:
        domain_path = self._validate_domain(domain)
        videos: dict[str, VideoSummary] = {}
        for caption_path in sorted(domain_path.glob("*.csv")):
            video_id = self._extract_video_id(caption_path.name)
            if video_id not in videos:
                segment_count = self._count_segments(caption_path)
                videos[video_id] = VideoSummary(
                    domain=domain,
                    video_id=video_id,
                    caption_path=str(caption_path),
                    segment_count=segment_count,
                )
        return list(videos.values())

    def load_video(self, domain: str, video_id: str) -> VideoSegmentsResponse:
        caption_path = self._resolve_caption_path(domain, video_id)
        segments = list(self._read_segments(caption_path))
        summary = VideoSummary(
            domain=domain,
            video_id=video_id,
            caption_path=str(caption_path),
            segment_count=len(segments),
        )
        return VideoSegmentsResponse(video=summary, segments=segments)

    def _validate_domain(self, domain: str) -> Path:
        domain_path = self.data_root / domain
        if not domain_path.exists() or not domain_path.is_dir():
            raise KeyError(f"Unknown domain: {domain}")
        return domain_path

    def _resolve_caption_path(self, domain: str, video_id: str) -> Path:
        domain_path = self._validate_domain(domain)
        candidates = sorted(domain_path.glob(f"{video_id}*.csv"))
        if not candidates:
            raise KeyError(f"Caption file not found for video '{video_id}' in domain '{domain}'")
        return candidates[0]

    @staticmethod
    def _extract_video_id(filename: str) -> str:
        return filename.split(".")[0]

    @staticmethod
    def _count_segments(caption_path: Path) -> int:
        with caption_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return sum(1 for _ in reader)

    def _read_segments(self, caption_path: Path) -> Iterable[Segment]:
        with caption_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                start = row.get("start") or row.get("\ufeffstart") or ""
                end = row.get("end") or ""
                text = row.get("text") or ""
                if not start or not end:
                    continue
                yield Segment(
                    segment_id=index,
                    start=start,
                    end=end,
                    start_sec=self._timestamp_to_seconds(start),
                    end_sec=self._timestamp_to_seconds(end),
                    text=text.strip(),
                )

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        hours, minutes, seconds = timestamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
