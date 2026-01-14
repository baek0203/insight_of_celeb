"""
extract_tfidf_important_sentences.py
클러스터 내에서 TF-IDF 점수가 높은 문장 추출
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def find_tfidf_important_sentences(csv_path, top_n=5):
    """
    클러스터 내에서 TF-IDF 점수가 높은 문장 추출
    """
    df = pd.read_csv(csv_path)
    df = df[df["cluster"] != -1].copy()
    
    important_sentences = []
    
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id].copy()
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_df["text"])
            
            # 각 문장의 TF-IDF 총합 (중요도 점수)
            scores = tfidf_matrix.sum(axis=1).A1
            cluster_df["tfidf_score"] = scores
            
            # 상위 문장 선택
            top_sentences = cluster_df.nlargest(top_n, "tfidf_score")
            
            for _, row in top_sentences.iterrows():
                important_sentences.append({
                    "cluster": cluster_id,
                    "video_id": row["source"].replace(".sent.en.csv", ""),
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                    "tfidf_score": row["tfidf_score"],
                    "importance": "high"
                })
        
        except Exception as e:
            print(f"⚠️ Cluster {cluster_id} TF-IDF failed: {e}")
    
    return pd.DataFrame(important_sentences)


df_important = find_tfidf_important_sentences(
    "outputs/step3/all_sentences_with_umap_commencement.csv",
    top_n=5
)

df_important.to_csv("outputs/step3/important_sentences_tfidf.csv", index=False)
