import os
import re
import torch
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ======================================================
# option
# ======================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index to use (default: 0)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate per text")
    parser.add_argument("--granularity", type=str, default="medium", choices=["fine", "medium", "coarse"],
                        help="Sentence splitting level: fine, medium, or coarse")
    parser.add_argument("--input-dir", type=str, default="data/clean/per_video/interview")
    parser.add_argument("--output-dir", type=str, default="data/split/interview_cleaned")
    return parser.parse_args()


# ======================================================
# load model
# ======================================================
def load_model(device: str, max_tokens: int, batch_size: int):
    MODEL_ID = "google/gemma-2-2b-it"
    print(f"Loading model: {MODEL_ID} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        max_new_tokens=max_tokens,
        do_sample=False,
        batch_size=batch_size
    )
    return generator


# ======================================================
# prompt
# ======================================================
# ======================================================
# prompt template (updated: remove filler words)
# ======================================================
def get_prompt(text: str, mode: str) -> str:
    if mode == "fine":
        instruction = (
            "Split the text into natural and complete English sentences. (8–20 words per sentence). "
            "If a sentence contains multiple ideas, divide it into multiple smaller ones."
        )
    elif mode == "coarse":
        instruction = (
            "Group related sentences together into short paragraphs that express one main idea."
        )
    else:
        instruction = (
            "Split the text into natural and complete English sentences. "
            "Combine short fragments with nearby sentences if they belong together."
        )

    return f"""
You are a sentence segmentation model.
{instruction}

Rules:
- Preserve all original words and punctuation unless explicitly told to remove them.
- Do NOT summarize, paraphrase, or rewrite meaning.
- Output one complete sentence per line.
- Do NOT include explanations or comments.
- Remove only meaningless filler words such as:
  ("uh", "um", "you know", "like", "ah", "er", "hmm", "so...", "well...", "I mean").
- Remove non-verbal interjections, laughter marks, or hesitation noises (e.g., "(laughs)", "(sigh)", "(pause)").
- Do NOT alter the semantics of the remaining sentence after removing filler words.

Text:
{text.strip()}

Output:
"""


# ======================================================
# batch parallel inference
# ======================================================
def split_sentences_batch(generator, texts: list[str], mode: str) -> list[list[str]]:
    prompts = [get_prompt(t, mode) for t in texts]
    outputs = generator(prompts)
    results = []

    for out_group in outputs:
        # pipeline이 [[{generated_text: ...}]] 구조로 반환됨
        out = out_group[0] if isinstance(out_group, list) else out_group
        result = out["generated_text"]

        if "Output:" in result:
            result = result.split("Output:")[-1]

        sentences = [
            line.strip()
            for line in result.splitlines()
            if line.strip() and not re.search(r"(sorry|let me|😊|thank|I can help)", line, re.I)
        ]
        results.append(sentences)
    return results


# ======================================================
# timestamp for original text
# ======================================================
# 🔧 수정: 문자열이 아닌 리스트를 반환하도록 변경
def attach_timestamps(start: str, end: str, sentences: list[str]) -> list[list]:
    # 각 문장마다 [start, end, sentence] 형태의 리스트로 반환
    return [[start, end, s] for s in sentences]


# ======================================================
# CSV 파일 처리 (Batch)
# ======================================================
def process_csv_batch(generator, input_csv: str, output_csv: str, batch_size: int, mode: str):
    df = pd.read_csv(input_csv)
    rows = []

    for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {os.path.basename(input_csv)}"):
        batch = df.iloc[i : i + batch_size]
        texts = batch["text"].tolist()
        starts = batch["start"].tolist()
        ends = batch["end"].tolist()

        split_results = split_sentences_batch(generator, texts, mode)
        for s, e, lines in zip(starts, ends, split_results):
            # 🔧 수정: extend로 리스트들을 추가 (각 리스트는 [start, end, text] 형태)
            rows.extend(attach_timestamps(s, e, lines))

    # 🔧 수정: rows는 이제 [[start, end, text], [start, end, text], ...] 형태
    out_df = pd.DataFrame(rows, columns=["start", "end", "text"]) 
    out_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ Saved to {output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    args = get_args()

    if torch.cuda.is_available():
        device_str = f"cuda:{args.gpu_id}"
        print(f"✅ Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device_str = "cpu"
        print("⚠️ CUDA not available, using CPU.")

    generator = load_model(device_str, args.max_tokens, args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_files = sorted(glob(os.path.join(args.input_dir, "*.en.csv")))

    for csv_path in csv_files:
        base = os.path.basename(csv_path)
        out_path = os.path.join(args.output_dir, base)
        process_csv_batch(generator, csv_path, out_path, args.batch_size, args.granularity)
