import os
import glob
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

ROOT = "data/clean/per_video"
SUBDIRS = ["startup"]  # 필요 시 확장
OUTPUT_ROOT = "data/split/startup_nltk"

# punkt 다운로드 보장 (punkt_tab 포함)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ===== 구두점 복원 모델 로드 (한 번만) =====
try:
    from deepmultilingualpunctuation import PunctuationModel
    punct_model = PunctuationModel()
    PUNCT_AVAILABLE = True
    print("✅ Punctuation restoration model loaded")
except ImportError:
    PUNCT_AVAILABLE = False
    print("⚠️  deepmultilingualpunctuation not installed. Install with: pip install deepmultilingualpunctuation")


def split_with_rules(text):
    """마침표/물음표/느낌표 뒤 무조건 분리 + 구두점 복원 + NLTK 조합"""
    text = text.strip()
    if not text:
        return []
    
    # 1단계: 마침표/물음표/느낌표 뒤 공백이 있으면 강제 분리
    parts = re.split(r'([.!?])\s+', text)
    
    temp_sents = []
    buffer = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:  # 텍스트 부분
            buffer += part
        else:  # 구두점 부분
            buffer += part
            if buffer.strip():
                temp_sents.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        temp_sents.append(buffer.strip())
    
    # 2단계: 30단어 이상 & 마침표 없으면 구두점 복원
    refined = []
    for s in temp_sents:
        words = s.split()
        # 마침표가 없고 30단어 이상이면 구두점 복원 시도
        if PUNCT_AVAILABLE and len(words) > 30 and not re.search(r'[.!?]$', s.strip()):
            try:
                restored = punct_model.restore_punctuation(s)
                # 복원된 결과 다시 분할
                sub_parts = re.split(r'([.!?])\s+', restored)
                sub_buffer = ""
                for j, p in enumerate(sub_parts):
                    if j % 2 == 0:
                        sub_buffer += p
                    else:
                        sub_buffer += p
                        if sub_buffer.strip():
                            refined.append(sub_buffer.strip())
                        sub_buffer = ""
                if sub_buffer.strip():
                    refined.append(sub_buffer.strip())
            except Exception as e:
                # 구두점 복원 실패 시 원본 사용
                print(f"⚠️  Punctuation restoration failed: {e}")
                refined.append(s)
        elif len(words) > 40:  # 40단어 이상만 NLTK 재분할
            refined.extend(sent_tokenize(s))
        else:
            refined.append(s)
    
    # 3단계: 접속사 재분할 (40단어 이상만)
    adjusted = []
    for s in refined:
        s = s.strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) > 40:
            cuts = []
            for i, tok in enumerate(toks):
                w = tok.lower().strip(",;:")
                if w in {"and", "but", "so", "because"} and i > 12 and len(toks) - i > 8:
                    cuts.append(i)
            if cuts:
                prev = 0
                c = cuts[0]
                left = " ".join(toks[prev:c]).strip()
                if left:
                    adjusted.append(left)
                right = " ".join(toks[c:]).strip()
                if right:
                    adjusted.append(right)
            else:
                adjusted.append(s)
        else:
            adjusted.append(s)
    
    # 4단계: 병합 (5단어 미만 또는 접속사 시작)
    out = []
    for s in adjusted:
        if not s:
            continue
        words = s.split()
        first = words[0].lower().strip('.,;:') if words else ''
        
        if (len(words) < 5 or first in {"and", "but", "so", "then", "because"}) and out:
            out[-1] = (out[-1] + " " + s).strip()
        else:
            out.append(s)
    
    return [s for s in out if s.strip()]


def redistribute_times(start, end, sents):
    """토큰 수 기반 가중 시간 분배"""
    weights = [max(1, len(word_tokenize(s))) for s in sents]
    total = sum(weights)
    dur = end - start
    acc = start
    spans = []
    
    for i, (s, w) in enumerate(zip(sents, weights)):
        if i < len(sents) - 1:
            seg = dur * (w / total) if total > 0 else dur / max(1, len(sents))
            spans.append((acc, acc + seg, s))
            acc += seg
        else:
            spans.append((acc, end, s))
    
    return spans


def to_seconds(hhmmss):
    """HH:MM:SS.mmm → float seconds"""
    hh, mm, ss = hhmmss.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def to_timecode(sec):
    """float seconds → HH:MM:SS.mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def process_csv(path):
    """하나의 CSV 파일을 문장 단위로 분할하고 시간 재분배"""
    df = pd.read_csv(path)
    
    # start, end를 초 단위 float로 변환
    def coerce(col):
        try:
            return df[col].astype(str).apply(to_seconds).astype(float)
        except Exception:
            return df[col].astype(float)
    
    starts = coerce("start")
    ends = coerce("end")

    out_rows = []
    for st, en, text in zip(starts, ends, df["text"].astype(str)):
        sents = split_with_rules(text)
        spans = redistribute_times(st, en, sents)
        for sst, een, sent in spans:
            out_rows.append({"start": sst, "end": een, "text": sent})

    out = pd.DataFrame(out_rows)
    
    # 타임코드 형식으로 복원
    out["start"] = out["start"].apply(to_timecode)
    out["end"] = out["end"].apply(to_timecode)

    # 출력 경로
    base = os.path.basename(path)
    save_name = base.replace(".en.csv", ".sent.en.csv")
    save_path = os.path.join(OUTPUT_ROOT, save_name)
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    return save_path, len(df), len(out)


def main():
    targets = []
    for sd in SUBDIRS:
        targets += glob.glob(os.path.join(ROOT, sd, "*.en.csv"))
    
    print(f"Found {len(targets)} files")

    stats = []
    for p in sorted(targets):
        save_path, n_in, n_out = process_csv(p)
        stats.append((p, save_path, n_in, n_out))
        print(f"{os.path.basename(p)} -> {os.path.basename(save_path)} | {n_in} rows → {n_out} rows")

    rep = pd.DataFrame(stats, columns=["input", "output", "rows_in", "rows_out"])
    rep_path = os.path.join(OUTPUT_ROOT, "sentence_split_report.csv")
    rep.to_csv(rep_path, index=False, encoding="utf-8-sig")
    print("Report written:", rep_path)


if __name__ == "__main__":
    main()
