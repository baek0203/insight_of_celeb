"""
download_videos_from_index.py
──────────────────────────────────────────────
subtitle_index.csv에 있는 영상 ID를 읽어서
YouTube에서 영상 파일(.mp4)을 다운로드합니다.

기존 collect_v2.py로 자막(VTT)만 받았다면,
이 스크립트로 영상을 추가로 다운로드할 수 있습니다.
"""

import os
import yt_dlp
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


def download_video_by_id(
    video_id: str,
    output_dir: str,
    quality: str = "best"
) -> bool:
    """
    단일 YouTube 영상 다운로드
    
    Args:
        video_id: YouTube 영상 ID
        output_dir: 저장 디렉토리
        quality: 화질 설정 ('best', '720p', '480p' 등)
    
    Returns:
        bool: 성공 여부
    """
    output_path = Path(output_dir) / f"{video_id}.mp4"
    
    # 이미 다운로드되어 있으면 스킵
    if output_path.exists():
        print(f"⏭️  Already exists: {video_id}")
        return True
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # 화질 설정
    if quality == "best":
        format_spec = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    elif quality == "720p":
        format_spec = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]"
    elif quality == "480p":
        format_spec = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]"
    else:
        format_spec = quality
    
    ydl_opts = {
        "format": format_spec,
        "outtmpl": str(output_path),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ Downloaded: {video_id} ({file_size:.1f} MB)")
            return True
        else:
            print(f"❌ Failed: {video_id} (file not created)")
            return False
    
    except Exception as e:
        print(f"❌ Error downloading {video_id}: {e}")
        return False


def download_videos_from_index(
    index_csv: str,
    output_dir: str = "data/raw/videos",
    quality: str = "best",
    max_videos: int = None
):
    """
    subtitle_index.csv에서 영상 ID를 읽어 다운로드
    
    Args:
        index_csv: subtitle_index.csv 경로
        output_dir: 영상 저장 디렉토리
        quality: 화질 설정
        max_videos: 최대 다운로드 개수 (None이면 전체)
    """
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 로드
    try:
        df = pd.read_csv(index_csv)
    except FileNotFoundError:
        print(f"❌ Index file not found: {index_csv}")
        return
    
    if "id" not in df.columns:
        print(f"❌ 'id' column not found in {index_csv}")
        return
    
    video_ids = df["id"].unique()
    
    if max_videos:
        video_ids = video_ids[:max_videos]
    
    print(f"\n📹 Starting download of {len(video_ids)} videos")
    print(f"   Quality: {quality}")
    print(f"   Output: {output_dir}\n")
    
    # 다운로드 통계
    start_time = datetime.now()
    success_count = 0
    failed_ids = []
    
    # 다운로드 실행
    for video_id in tqdm(video_ids, desc="Downloading"):
        if download_video_by_id(video_id, output_dir, quality):
            success_count += 1
        else:
            failed_ids.append(video_id)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ Successfully downloaded: {success_count}/{len(video_ids)}")
    
    if failed_ids:
        print(f"❌ Failed videos: {len(failed_ids)}")
        print(f"   IDs: {', '.join(failed_ids[:10])}")
        if len(failed_ids) > 10:
            print(f"   ... and {len(failed_ids) - 10} more")
    
    print(f"⏱  Total time: {datetime.now() - start_time}")
    print(f"📁 Videos saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    # 실패 목록 저장 (선택)
    if failed_ids:
        failed_csv = Path(output_dir) / "failed_downloads.csv"
        pd.DataFrame({"video_id": failed_ids}).to_csv(failed_csv, index=False)
        print(f"📄 Failed IDs saved to: {failed_csv}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download YouTube videos from subtitle_index.csv"
    )
    parser.add_argument(
        "--index",
        default="data/raw/commencement/subtitle_index.csv",
        help="Path to subtitle_index.csv"
    )
    parser.add_argument(
        "--output",
        default="data/raw/videos",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--quality",
        default="best",
        choices=["best", "720p", "480p"],
        help="Video quality"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of videos to download"
    )
    
    args = parser.parse_args()
    
    download_videos_from_index(
        index_csv=args.index,
        output_dir=args.output,
        quality=args.quality,
        max_videos=args.max
    )


if __name__ == "__main__":
    main()
