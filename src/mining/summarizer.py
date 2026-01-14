import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from tqdm import tqdm

# ======================================================
# 0️⃣ 디렉토리 설정
# ======================================================
OUTPUT_DIR = "outputs/step3/core_keyword_analysis_startup_llm"
WORDCLOUD_DIR = os.path.join(OUTPUT_DIR, "wordclouds")
KEYWORD_DIR = os.path.join(OUTPUT_DIR, "keywords")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORDCLOUD_DIR, exist_ok=True)
os.makedirs(KEYWORD_DIR, exist_ok=True)

print(f"✅ Output directories created:")
print(f"   - Main: {OUTPUT_DIR}")
print(f"   - Wordclouds: {WORDCLOUD_DIR}")
print(f"   - Keywords: {KEYWORD_DIR}\n")

# 🔧 추가: 구어체 및 불필요 단어 리스트
CUSTOM_STOPWORDS = {
    # 기본 불용어
    'like', 'know', 'just', 'think', 'really', 'going', 'yeah', 'want', 'got', 'getting',
    # 구어체 삽입어
    'uh', 'um', 'ah', 'er', 'hmm', 'mm', 'oh',
    # 일반적인 단어
    'thing', 'things', 'people', 'time', 'way', 'right', 'kind', 've', 'll', 'don',
    # 대화 관련
    'say', 'said', 'tell', 'told', 'talk', 'talking', 'mean', 'feel', 'feeling',
    # 기타 일반어
    'good', 'bad', 'better', 'best', 'lot', 'little', 'big', 'small',
    'day', 'year', 'years', 'today', 'didn', 'doesn', 'wasn', 'weren',
    # 불완전 단어
    'gt', 'gt gt', 'thank', 'thanks'
}

# ======================================================
# 1️⃣ 모델 로드
# ======================================================
def load_model():
    MODEL_ID = "google/gemma-2-9b-it"
    print(f"🚀 Loading model: {MODEL_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda:0")
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda:0"),
        max_new_tokens=300,
        do_sample=False
    )
    return generator

# ======================================================
# 2️⃣ 개선된 TF-IDF 키워드 추출
# ======================================================
def extract_keywords_tfidf(texts, top_n=30):
    """TF-IDF를 사용하여 의미있는 키워드 추출"""
    try:
        # 기본 영어 불용어 + 커스텀 불용어 결합
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        all_stopwords = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))
        
        tfidf = TfidfVectorizer(
            stop_words=all_stopwords,
            max_features=3000,
            min_df=3,  # 최소 3개 문서에 나타나야 함
            max_df=0.7,  # 70% 이상 문서에 나타나는 단어 제외
            ngram_range=(1, 3),  # 1-gram부터 3-gram까지
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # 최소 3글자 이상 단어만
        )
        
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = np.array(tfidf.get_feature_names_out())
        scores = tfidf_matrix.mean(axis=0).A1
        
        # 🔧 추가 필터링: 숫자만 있는 단어 제거
        valid_indices = []
        for idx, word in enumerate(feature_names):
            # 숫자만 있거나 너무 짧은 단어 제외
            if not word.replace(' ', '').isdigit() and len(word.strip()) >= 3:
                valid_indices.append(idx)
        
        if len(valid_indices) == 0:
            return []
        
        feature_names = feature_names[valid_indices]
        scores = scores[valid_indices]
        
        # 상위 키워드 추출
        top_indices = scores.argsort()[-top_n:][::-1]
        top_keywords = feature_names[top_indices]
        top_scores = scores[top_indices]
        
        return list(zip(top_keywords, top_scores))
    except Exception as e:
        print(f"   ⚠️ TF-IDF failed: {e}")
        return []

# ======================================================
# 3️⃣ 워드클라우드 생성
# ======================================================
def create_wordcloud(keywords_with_scores, cluster_id, cluster_size):
    """키워드로 워드클라우드 생성"""
    if len(keywords_with_scores) == 0:
        return None
    
    word_freq = dict(keywords_with_scores)
    
    wc = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Cluster {cluster_id} Key Topics ({cluster_size} sentences)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(WORDCLOUD_DIR, f'cluster_{cluster_id}_wordcloud.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path

# ======================================================
# 4️⃣ 키워드 바 차트 생성
# ======================================================
def create_keyword_chart(keywords_with_scores, cluster_id, cluster_size):
    """상위 키워드를 바 차트로 시각화"""
    if len(keywords_with_scores) == 0:
        return None
    
    # 상위 15개만 선택
    top_15 = keywords_with_scores[:15]
    keywords = [kw for kw, _ in top_15]
    scores = [score for _, score in top_15]
    
    # 바 차트 생성
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(keywords)), scores, color='steelblue', alpha=0.8)
    plt.yticks(range(len(keywords)), keywords, fontsize=11)
    plt.xlabel('TF-IDF Score', fontsize=12)
    plt.title(f'Cluster {cluster_id} Top Topics ({cluster_size} sentences)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    
    # 값 표시
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score, i, f' {score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    output_path = os.path.join(KEYWORD_DIR, f'cluster_{cluster_id}_keywords.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path

# ======================================================
# 5️⃣ 클러스터 주제 및 가치관 분석
# ======================================================
def analyze_cluster_theme(generator, texts, cluster_id, max_samples=20):
    """LLM을 사용한 주제 및 가치관 분석"""
    if len(texts) > max_samples:
        sample_texts = np.random.choice(texts, max_samples, replace=False)
    else:
        sample_texts = texts
    
    sentences_list = "\n".join([f"{i+1}. {t[:200]}" for i, t in enumerate(sample_texts)])
    
    prompt = f"""You are an expert in thematic analysis and value extraction.

Analyze the following sentences from Cluster {cluster_id} and provide:

1. **Main Theme**: What is the central topic or subject matter? (1-2 sentences)
2. **Core Values**: What values, beliefs, or perspectives are expressed? (1-2 sentences)
3. **Summary**: Summarize the key message in simple terms (2-3 sentences)

Sentences:
{sentences_list}

Analysis:
1. Main Theme:"""
    
    output = generator(prompt, max_new_tokens=300)[0]["generated_text"]
    
    if "Analysis:" in output:
        analysis = output.split("Analysis:")[-1].strip()
    else:
        analysis = output.split(prompt)[-1].strip() if prompt in output else output
    
    return analysis

# ======================================================
# 6️⃣ 메인 코드
# ======================================================
print("🔹 Loading data...")
df = pd.read_csv("outputs/step3/all_sentences_with_umap_startup_llm.csv")
df = df.dropna(subset=["text"])

n_clusters = len(df["cluster"].unique()) - (1 if -1 in df["cluster"].unique() else 0)
print(f"📊 Found {n_clusters} clusters in the data\n")

# 모델 로드
generator = load_model()

# 결과 저장용
cluster_analyses = []

print("\n🔹 Analyzing clusters...\n")
for cid in tqdm(sorted(df["cluster"].unique()), desc="Processing clusters"):
    if cid == -1: 
        continue
    
    texts = df[df["cluster"] == cid]["text"].tolist()
    
    if len(texts) < 5:
        print(f"\n⚠️ Cluster {cid}: Only {len(texts)} sentences, skipping...")
        continue
    
    print(f"\n{'='*60}")
    print(f"📊 Cluster {cid} ({len(texts)} sentences)")
    print(f"{'='*60}")
    
    # 대표 문장 출력
    print("\n📝 Sample sentences:")
    for i, sent in enumerate(texts[:3], 1):
        print(f"   {i}. {sent[:150]}...")
    
    # 키워드 추출
    print("\n🔹 Extracting meaningful keywords...")
    keywords_with_scores = extract_keywords_tfidf(texts, top_n=30)
    
    if len(keywords_with_scores) == 0:
        print("   ⚠️ No meaningful keywords found")
        top_10_keywords = []
    else:
        top_10_keywords = [kw for kw, _ in keywords_with_scores[:10]]
        print(f"   Top 10 topics: {', '.join(top_10_keywords)}")
    
    # 워드클라우드 생성
    wordcloud_path = create_wordcloud(keywords_with_scores, cid, len(texts))
    if wordcloud_path:
        print(f"   ✅ Wordcloud saved: {os.path.basename(wordcloud_path)}")
    
    # 키워드 차트 생성
    keyword_chart_path = create_keyword_chart(keywords_with_scores, cid, len(texts))
    if keyword_chart_path:
        print(f"   ✅ Keyword chart saved: {os.path.basename(keyword_chart_path)}")
    
    # LLM 분석
    print("\n🔹 Analyzing theme and values...")
    analysis = analyze_cluster_theme(generator, texts, cid, max_samples=20)
    
    print(f"\n🔍 Analysis:\n{analysis}\n")
    
    # 결과 저장
    cluster_analyses.append({
        "cluster_id": cid,
        "size": len(texts),
        "top_keywords": ", ".join(top_10_keywords) if top_10_keywords else "N/A",
        "sample_sentences": " | ".join(texts[:3]),
        "analysis": analysis,
        "wordcloud_path": wordcloud_path or "",
        "keyword_chart_path": keyword_chart_path or ""
    })
    
    # 개별 텍스트 파일 저장
    with open(os.path.join(OUTPUT_DIR, f"cluster_{cid}_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(f"Cluster {cid} Analysis\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Size: {len(texts)} sentences\n\n")
        
        if top_10_keywords:
            f.write(f"Top 10 Topics:\n")
            f.write(f"{', '.join(top_10_keywords)}\n\n")
        
        f.write(f"Sample Sentences:\n")
        for i, sent in enumerate(texts[:3], 1):
            f.write(f"{i}. {sent}\n\n")
        
        f.write(f"\nTheme & Values Analysis:\n{analysis}\n")

# ======================================================
# 7️⃣ 결과 저장
# ======================================================
result_df = pd.DataFrame(cluster_analyses)
csv_path = os.path.join(OUTPUT_DIR, "cluster_themes_and_values.csv")
result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n{'='*60}")
print(f"✅ Analysis complete!")
print(f"✅ Results saved to: {csv_path}")
print(f"✅ Individual analyses: {OUTPUT_DIR}/cluster_*_analysis.txt")
print(f"✅ Wordclouds: {WORDCLOUD_DIR}/")
print(f"✅ Keyword charts: {KEYWORD_DIR}/")
print(f"{'='*60}\n")

# 요약 테이블 출력
print(f"📋 Summary Table ({len(result_df)} clusters):")
for _, row in result_df.iterrows():
    print(f"\n   Cluster {row['cluster_id']:2d} ({row['size']:3d} sentences):")
    print(f"   Topics: {row['top_keywords'][:80]}{'...' if len(row['top_keywords']) > 80 else ''}")

# 전체 요약 파일 생성
summary_path = os.path.join(OUTPUT_DIR, "all_clusters_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(f"ALL CLUSTERS ANALYSIS SUMMARY ({len(result_df)} clusters)\n")
    f.write("=" * 80 + "\n\n")
    
    for _, row in result_df.iterrows():
        f.write(f"\n{'='*60}\n")
        f.write(f"Cluster {row['cluster_id']} ({row['size']} sentences)\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Top Topics: {row['top_keywords']}\n\n")
        f.write(f"Analysis:\n{row['analysis']}\n\n")

print(f"\n✅ Summary file saved to: {summary_path}")

# 키워드 통계
print(f"\n💡 Analysis Statistics:")
print(f"   - Total clusters analyzed: {len(result_df)}")
print(f"   - Wordclouds generated: {len([p for p in result_df['wordcloud_path'] if p])}")
print(f"   - Keyword charts generated: {len([p for p in result_df['keyword_chart_path'] if p])}")
