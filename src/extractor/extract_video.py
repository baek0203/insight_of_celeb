"""
extract_representative_sentences.py
각 클러스터에서 중심에 가장 가까운 대표 문장 추출
"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean


def find_cluster_representatives(csv_path, top_n=5):
    """
    각 클러스터의 대표 문장 추출
    
    Args:
        csv_path: UMAP + cluster 정보가 있는 CSV
        top_n: 클러스터당 추출할 문장 수
    
    Returns:
        DataFrame with representative sentences
    """
    df = pd.read_csv(csv_path)
    
    # 노이즈 제거 (cluster == -1)
    df = df[df["cluster"] != -1].copy()
    
    representatives = []
    
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id].copy()
        
        # 클러스터 중심 계산
        center_x = cluster_df["UMAP_1"].mean()
        center_y = cluster_df["UMAP_2"].mean()
        
        # 각 문장과 중심 간 거리 계산
        cluster_df["distance_to_center"] = cluster_df.apply(
            lambda row: euclidean([row["UMAP_1"], row["UMAP_2"]], [center_x, center_y]),
            axis=1
        )
        
        # 중심에 가까운 순으로 정렬
        top_sentences = cluster_df.nsmallest(top_n, "distance_to_center")
        
        for _, row in top_sentences.iterrows():
            representatives.append({
                "cluster": cluster_id,
                "video_id": row["source"].replace(".sent.en.csv", ""),
                "start": row["start"],
                "end": row["end"],
                "text": row["text"],
                "distance_to_center": row["distance_to_center"],
                "representativeness": "high"  # 중심에 가까우므로
            })
        
        print(f"Cluster {cluster_id}: Selected {len(top_sentences)} representative sentences")
    
    return pd.DataFrame(representatives)


# 실행
df_reps = find_cluster_representatives(
    "outputs/step3/all_sentences_with_umap_commencement.csv",
    top_n=5
)

# 저장
df_reps.to_csv("outputs/step3/representative_sentences.csv", index=False)

# 샘플 출력
print("\n📌 Sample Representatives:")
for cid in [0, 5, 11]:  # 주요 클러스터만
    print(f"\n--- Cluster {cid} ---")
    samples = df_reps[df_reps["cluster"] == cid].head(2)
    for _, row in samples.iterrows():
        print(f"  [{row['start']}] {row['text'][:80]}...")
