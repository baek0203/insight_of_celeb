"""
video_utils.py
영상 추출 유틸리티 함수 모음
"""
import subprocess
from pathlib import Path
import pandas as pd


def extract_clip(input_video, start_time, end_time, output_path):
    """
    ffmpeg로 영상 구간 추출
    
    Args:
        input_video: 원본 영상 경로
        start_time: 시작 시간 (HH:MM:SS.mmm)
        end_time: 종료 시간 (HH:MM:SS.mmm)
        output_path: 출력 파일 경로
    
    Returns:
        bool: 성공 여부
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-ss", start_time,
        "-to", end_time,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-avoid_negative_ts", "1",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def concat_videos(clip_paths, output_path):
    """
    여러 클립을 하나로 병합
    
    Args:
        clip_paths: 클립 경로 리스트
        output_path: 출력 파일 경로
    """
    filelist_path = output_path.parent / "filelist.txt"
    
    with open(filelist_path, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{Path(clip).absolute()}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(filelist_path),
        "-c", "copy",
        str(output_path)
    ]
    
    subprocess.run(cmd)
    filelist_path.unlink()  # 임시 파일 삭제


def add_text_overlay(input_video, text, output_path, duration=None):
    """
    영상에 텍스트 오버레이 추가
    
    Args:
        input_video: 입력 영상
        text: 표시할 텍스트
        output_path: 출력 경로
        duration: 텍스트 표시 시간 (None이면 전체)
    """
    # 특수문자 이스케이프
    text = text.replace("'", "'\\''").replace(":", "\\:")
    
    if duration:
        drawtext = f"drawtext=text='{text}':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=h-80:enable='between(t,0,{duration})'"
    else:
        drawtext = f"drawtext=text='{text}':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=h-80"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", drawtext,
        "-c:a", "copy",
        str(output_path)
    ]
    
    subprocess.run(cmd)


def get_video_duration(video_path):
    """영상 길이 가져오기"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return None
