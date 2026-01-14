"""
YouTube Subtitle Collector
──────────────────────────────────────────────
Download English subtitles from YouTube playlists.
Prioritizes auto-generated captions when available.
"""

import argparse
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yt_dlp


def download_playlist_subtitles(
    playlist_url: str,
    domain_name: str = "interview",
    base_dir: str = "data/raw",
    languages: Optional[List[str]] = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Download subtitles from a YouTube playlist.

    Args:
        playlist_url: YouTube playlist URL
        domain_name: Domain name for organizing outputs
        base_dir: Base directory for saving subtitles
        languages: List of language codes (default: ["en"])
        quiet: Suppress yt-dlp output

    Returns:
        DataFrame with downloaded subtitle information
    """
    if languages is None:
        languages = ["en"]

    save_dir = os.path.join(base_dir, domain_name)
    os.makedirs(save_dir, exist_ok=True)
    index_csv = os.path.join(save_dir, "subtitle_index.csv")

    ydl_opts = {
        "writeautomaticsub": True,
        "subtitleslangs": languages,
        "skip_download": True,
        "outtmpl": f"{save_dir}/%(id)s.%(ext)s",
        "extract_flat": False,
        "quiet": quiet,
    }

    records = []
    start_time = datetime.now()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
            entries = info.get("entries", [])
            print(f"\n📋 재생목록: {info.get('title', 'Unknown')} ({len(entries)}개 영상)\n")

            for entry in entries:
                video_id = entry.get("id")
                title = entry.get("title", "")
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                print(f"▶ {title[:70]}...")

                try:
                    ydl.download([video_url])
                    vtt_path = os.path.join(save_dir, f"{video_id}.en.vtt")

                    if os.path.exists(vtt_path):
                        print(f"   ✅ 자막 다운로드 완료: {vtt_path}")
                        records.append({
                            "id": video_id,
                            "title": title,
                            "url": video_url,
                            "path": vtt_path,
                        })
                    else:
                        print("   ⚠️ 자막이 존재하지 않음 (auto captions 없음)")

                except Exception as e:
                    print(f"   ❌ 실패: {e}")

        except Exception as e:
            print(f"\n❌ 재생목록 다운로드 실패: {e}")
            return

    # Save index CSV
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(index_csv, index=False, encoding="utf-8-sig")
        print(f"\n Subtitle index saved: {index_csv} ({len(records)} files)")
    else:
        print("\n No subtitles downloaded.")

    print(f"\n Total time: {datetime.now() - start_time}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube playlist subtitles")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--domain", default="interview", help="Domain name")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--lang", default="en", help="Subtitle language")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()
    download_playlist_subtitles(
        args.url,
        domain_name=args.domain,
        base_dir=args.output,
        languages=[args.lang],
        quiet=args.quiet,
    )