import shap
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =======================================================
# 1️⃣ 데이터 준비
# =======================================================
df = pd.read_csv("outputs/step3/all_sentences_with_umap_startup.csv")
texts = df["text"].tolist()
labels = df["cluster"].astype(int).tolist()

# =======================================================
# 2️⃣ 문장 임베딩
# =======================================================
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# =======================================================
# 3️⃣ 분류 모델 (클러스터 예측용)
# =======================================================
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
clf.fit(X_train, y_train)

# =======================================================
# 4️⃣ SHAP 적용
# =======================================================
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_test[:100])  # 상위 100개 문장만 예시

# 전역 feature 중요도 시각화
shap.summary_plot(shap_values, X_test[:100], feature_names=[f"dim_{i}" for i in range(embeddings.shape[1])])

from transformers import pipeline

# 1️⃣ 텍스트 분류기 파이프라인 정의 (LLM or fine-tuned model)
classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

# 2️⃣ SHAP Explainer 적용
explainer = shap.Explainer(classifier)
shap_values = explainer(["AI agents will transform the world of startups."])

# 3️⃣ 시각화
shap.plots.text(shap_values[0])

