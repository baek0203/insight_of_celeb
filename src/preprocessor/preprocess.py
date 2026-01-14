"""
preprocess.py (시간 윈도우 병합 버전)
──────────────────────────────────────────────
improvements:
- 20-30초 단위로 자막 병합
- 중복 텍스트 제거 (순차적)
- 완전한 문맥 보존
"""

import webvtt
import os
import pandas as pd
from tqdm import tqdm
import re


RAW_ROOT = "data/raw"
CLEAN_ROOT = "data/clean/startup/"
LOG_ROOT = "data/clean/logs"


# ────────────────────────────────
# 로그 관리 (기존 유지)
# ────────────────────────────────
def get_processed_ids(domain):
    """이미 처리된 영상 목록 불러오기 (도메인별 로그)"""
    log_path = os.path.join(LOG_ROOT, f"processed_log_{domain}.csv")
    if os.path.exists(log_path):
        return set(pd.read_csv(log_path)["video_id"].tolist())
    return set()


def append_processed_id(domain, video_id):
    """처리 완료된 영상 ID를 로그에 추가"""
    os.makedirs(LOG_ROOT, exist_ok=True)
    log_path = os.path.join(LOG_ROOT, f"processed_log_{domain}.csv")
    df = pd.DataFrame({"video_id": [video_id]})
    mode = "a" if os.path.exists(log_path) else "w"
    header = not os.path.exists(log_path)
    df.to_csv(log_path, mode=mode, header=header, index=False)


# ────────────────────────────────
# utils
# ────────────────────────────────
def timestamp_to_seconds(timestamp):
    """00:01:23.456 → 83.456 초"""
    parts = timestamp.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def clean_text(text: str):
    """텍스트 정제"""
    # [Music], [Applause] 등 제거
    text = re.sub(r'\[.*?\]', '', text)
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ────────────────────────────────
# VTT 파싱 (개선)
# ────────────────────────────────
def parse_vtt_file(vtt_path: str):
    """단일 VTT 파일을 DataFrame 형태로 변환"""
    video_id = os.path.basename(vtt_path).split(".")[0]
    records = []
    
    try:
        for caption in webvtt.read(vtt_path):
            text = caption.text.strip().replace('\n', ' ')
            if len(text.split()) >= 2:
                records.append({
                    "video_id": video_id,
                    "start": caption.start,
                    "end": caption.end,
                    "start_sec": timestamp_to_seconds(caption.start),
                    "end_sec": timestamp_to_seconds(caption.end),
                    "text": text
                })
    except Exception as e:
        print(f"⚠️ {vtt_path} 파싱 실패: {e}")
    
    return pd.DataFrame(records)


# ────────────────────────────────
# 중복 제거 로직 (개선)
# ────────────────────────────────
def get_overlap_prefix(prev_words, curr_words):
    """이전 행과 현재 행 간 중복 구간 탐색"""
    max_overlap = 0
    min_len = min(len(prev_words), len(curr_words))
    for i in range(1, min_len + 1):
        if prev_words[-i:] == curr_words[:i]:
            max_overlap = i
    return max_overlap


def remove_sequential_overlap(texts):
    """
    순차적으로 여러 텍스트의 중복 제거 및 병합
    
    Input:  ["at some point you have", "you have to believe", "believe something"]
    Output: "at some point you have to believe something"
    """
    if not texts:
        return ""
    
    result_words = texts[0].lower().split()
    
    for i in range(1, len(texts)):
        curr_words = texts[i].lower().split()
        
        # 이전 결과와 현재 텍스트의 중복 찾기
        overlap = get_overlap_prefix(result_words, curr_words)
        
        # 중복 제거 후 추가
        new_words = curr_words[overlap:] if overlap > 0 else curr_words
        result_words.extend(new_words)
    
    return ' '.join(result_words)


# ────────────────────────────────
# 시간 윈도우 병합 (핵심 추가)
# ────────────────────────────────
def merge_by_time_window(df: pd.DataFrame, window_seconds=25):
    """
    시간 윈도우 단위로 자막 병합
    
    Args:
        df: 원본 자막 DataFrame
        window_seconds: 병합 시간 윈도우 (초)
    
    Returns:
        병합된 DataFrame (start, end, text)
    """
    if df.empty:
        return pd.DataFrame()
    
    merged_segments = []
    current_segment = {
        'start': df.iloc[0]['start'],
        'start_sec': df.iloc[0]['start_sec'],
        'end': df.iloc[0]['end'],
        'end_sec': df.iloc[0]['end_sec'],
        'texts': [df.iloc[0]['text']]
    }
    
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        
        # 현재 세그먼트의 길이 계산
        segment_duration = row['start_sec'] - current_segment['start_sec']
        
        # Pause 길이 계산
        pause = row['start_sec'] - current_segment['end_sec']
        
        # 병합 종료 조건
        # 1. 시간 윈도우 초과
        # 2. 긴 pause (3초 이상)
        if segment_duration >= window_seconds or pause > 3.0:
            # 현재 세그먼트 저장
            merged_text = remove_sequential_overlap(current_segment['texts'])
            merged_text = clean_text(merged_text)
            
            # 최소 10단어 이상만 저장
            if merged_text and len(merged_text.split()) >= 10:
                merged_segments.append({
                    'start': current_segment['start'],
                    'end': current_segment['end'],
                    'text': merged_text
                })
            
            # 새 세그먼트 시작
            current_segment = {
                'start': row['start'],
                'start_sec': row['start_sec'],
                'end': row['end'],
                'end_sec': row['end_sec'],
                'texts': [row['text']]
            }
        else:
            # 현재 세그먼트에 추가
            current_segment['texts'].append(row['text'])
            current_segment['end'] = row['end']
            current_segment['end_sec'] = row['end_sec']
    
    # 마지막 세그먼트 처리
    if current_segment['texts']:
        merged_text = remove_sequential_overlap(current_segment['texts'])
        merged_text = clean_text(merged_text)
        
        if merged_text and len(merged_text.split()) >= 10:
            merged_segments.append({
                'start': current_segment['start'],
                'end': current_segment['end'],
                'text': merged_text
            })
    
    return pd.DataFrame(merged_segments)


# ────────────────────────────────
# 파일 단위 처리 (개선)
# ────────────────────────────────
def process_single_video(domain, vtt_path, window_seconds=25):
    """단일 영상 전처리 (시간 윈도우 병합)"""
    vid = os.path.splitext(os.path.basename(vtt_path))[0]
    print(f"\n🎬 [{domain}] {vid} 처리 중...")
    
    # 1. VTT 파싱
    df = parse_vtt_file(vtt_path)
    if df.empty:
        print(f"⚠️ {vid}: 자막 비어 있음, 건너뜀")
        return
    
    print(f"   📄 원본: {len(df)}줄")
    
    # 2. 시간 윈도우 병합
    df_merged = merge_by_time_window(df, window_seconds=window_seconds)
    
    if df_merged.empty:
        print(f"⚠️ {vid}: 병합 후 데이터 없음, 건너뜀")
        return
    
    print(f"   ⏱  {window_seconds}초 병합: {len(df)}줄 → {len(df_merged)}세그먼트")
    
    # 3. 최종 저장
    out_dir = os.path.join(CLEAN_ROOT, domain)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{vid}.csv")
    
    df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    append_processed_id(domain, vid)
    print(f"✅ 저장: {out_path} ({len(df_merged)}세그먼트)")


# ────────────────────────────────
# 도메인 단위 처리 (개선)
# ────────────────────────────────
def process_domain(domain, window_seconds=25):
    """하나의 도메인 내 모든 영상 처리"""
    domain_raw_dir = os.path.join(RAW_ROOT, domain)
    if not os.path.exists(domain_raw_dir):
        print(f"⚠️ {domain_raw_dir} 폴더가 없습니다. 건너뜀.")
        return
    
    vtts = [f for f in os.listdir(domain_raw_dir) if f.endswith(".vtt")]
    processed = get_processed_ids(domain)
    new_vtts = [f for f in vtts if os.path.splitext(f)[0] not in processed]
    
    if not new_vtts:
        print(f"✨ [{domain}] 모든 영상이 최신 상태입니다.")
        return
    
    print(f"\n📂 도메인: {domain} ({len(new_vtts)}개 신규)")
    print(f"⏱  시간 윈도우: {window_seconds}초")
    
    for vtt in new_vtts:
        process_single_video(domain, os.path.join(domain_raw_dir, vtt), 
                           window_seconds=window_seconds)


def process_all_domains(window_seconds=25):
    """모든 도메인 폴더 순회"""
    domains = [d for d in os.listdir(RAW_ROOT) 
               if os.path.isdir(os.path.join(RAW_ROOT, d))]
    
    if not domains:
        print("❌ 처리할 도메인 폴더가 없습니다.")
        return
    
    print("="*80)
    print(f"자막 전처리 시작 (시간 윈도우: {window_seconds}초)")
    print("="*80)
    
    for domain in domains:
        process_domain(domain, window_seconds=window_seconds)
    
    print("\n" + "="*80)
    print("✅ 전처리 완료!")
    print("="*80)


# ────────────────────────────────
# 실행부
# ────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="자막 전처리 (시간 윈도우 병합)"
    )
    parser.add_argument(
        "--window", 
        type=int, 
        default=25,
        help="병합 시간 윈도우 (초), 기본 25초"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="특정 도메인만 처리"
    )
    args = parser.parse_args()
    
    if args.domain:
        process_domain(args.domain, window_seconds=args.window)
    else:
        process_all_domains(window_seconds=args.window)
