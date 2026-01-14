import os
import pandas as pd
from glob import glob

input_dir = "data/clean/per_video/startup"
output_root = "data/split/startup"

os.makedirs(output_root, exist_ok=True)

# 모든 csv 파일 읽기
csv_files = sorted(glob(os.path.join(input_dir, "*.en.csv")))

for file_path in csv_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(output_root, base_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)

    # 행단위로 분리 저장
    for idx, row in df.iterrows():
        row_df = pd.DataFrame([row])
        out_path = os.path.join(output_dir, f"row_{idx:03d}.csv")
        row_df.to_csv(out_path, index=False)

    print(f"✅ {base_name}: {len(df)} rows saved to {output_dir}")
