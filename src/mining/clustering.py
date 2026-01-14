"""
Text Clustering Pipeline
──────────────────────────────────────────────
Cluster text data using UMAP + HDBSCAN with automatic cluster merging.
"""

import argparse
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class ClusteringConfig:
    """Configuration for clustering pipeline."""

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cuda"

    # UMAP
    umap_n_neighbors: int = 25
    umap_n_components: int = 2
    umap_metric: str = "cosine"
    umap_random_state: int = 42

    # HDBSCAN
    hdbscan_min_cluster_size: int = 12
    hdbscan_min_samples: int = 5
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"

    # Cluster merging
    target_min_clusters: int = 15
    target_max_clusters: int = 20
    merge_small_threshold: int = 10
    merge_medium_threshold: int = 25
    merge_large_threshold: int = 35

    # Noise handling
    noise_reassign_threshold: float = 0.10


@dataclass
class ClusteringResult:
    """Result of clustering pipeline."""

    df_all: pd.DataFrame
    embeddings: np.ndarray
    reduced: np.ndarray
    cluster_labels: np.ndarray
    representatives: pd.DataFrame
    n_clusters: int
    n_noise: int


class ClusteringPipeline:
    """Text clustering pipeline using UMAP + HDBSCAN."""

    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.model = None
        self.reducer = None
        self.clusterer = None

    def load_data(self, input_dir: str, text_column: str = "text") -> Tuple[pd.DataFrame, List[str]]:
        """Load CSV files from directory."""
        csv_files = sorted(glob(os.path.join(input_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")

        all_rows = []
        for f in csv_files:
            df = pd.read_csv(f)
            df["source"] = os.path.basename(f)
            all_rows.append(df)

        df_all = pd.concat(all_rows, ignore_index=True)
        sentences = df_all[text_column].astype(str).tolist()
        print(f"Loaded {len(sentences):,} sentences from {len(csv_files)} files")

        return df_all, sentences

    def encode(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences using SentenceTransformer."""
        print(f"\nEncoding sentences with {self.config.embedding_model}...")

        if self.model is None:
            self.model = SentenceTransformer(self.config.embedding_model)

        embeddings = self.model.encode(
            sentences,
            show_progress_bar=True,
            device=self.config.device
        )
        return embeddings

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensions using UMAP."""
        print("\nReducing dimensions with UMAP...")

        self.reducer = umap.UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            metric=self.config.umap_metric,
            random_state=self.config.umap_random_state
        )
        reduced = self.reducer.fit_transform(embeddings)
        return reduced

    def cluster(self, reduced: np.ndarray) -> np.ndarray:
        """Cluster using HDBSCAN."""
        print("\nClustering with HDBSCAN...")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric=self.config.hdbscan_metric,
            cluster_selection_method=self.config.hdbscan_cluster_selection_method
        )
        cluster_labels = self.clusterer.fit_predict(reduced)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Initial clustering: {n_clusters} clusters")

        return cluster_labels

    def merge_clusters(
        self,
        cluster_labels: np.ndarray,
        reduced: np.ndarray
    ) -> np.ndarray:
        """Merge small clusters to reach target count."""
        cfg = self.config
        labels = cluster_labels.copy()

        def get_cluster_sizes() -> Dict[int, int]:
            return {cid: (labels == cid).sum() for cid in set(labels) if cid != -1}

        def merge_small(threshold: int, max_merges: int = None):
            nonlocal labels
            cluster_sizes = get_cluster_sizes()
            small = [cid for cid, size in cluster_sizes.items() if size < threshold]

            if max_merges:
                small = sorted(small, key=lambda c: cluster_sizes[c])[:max_merges]

            for small_cid in small:
                small_mask = labels == small_cid
                small_centroid = reduced[small_mask].mean(axis=0)

                min_dist = float('inf')
                nearest_cid = None
                for other_cid in cluster_sizes.keys():
                    if other_cid == small_cid:
                        continue
                    other_centroid = reduced[labels == other_cid].mean(axis=0)
                    dist = np.linalg.norm(small_centroid - other_centroid)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cid = other_cid

                if nearest_cid is not None:
                    labels[small_mask] = nearest_cid

        # Step 1: Merge very small clusters
        print("\nMerging small clusters...")
        while True:
            sizes = get_cluster_sizes()
            very_small = [c for c, s in sizes.items() if s < cfg.merge_small_threshold]
            if not very_small:
                break
            merge_small(cfg.merge_small_threshold)

        current_n = len(get_cluster_sizes())
        print(f"  After small merge: {current_n} clusters")

        # Step 2: Merge medium clusters
        if current_n > cfg.target_max_clusters:
            for _ in range(10):
                if len(get_cluster_sizes()) <= cfg.target_max_clusters:
                    break
                merge_small(cfg.merge_medium_threshold, max_merges=5)

            current_n = len(get_cluster_sizes())
            print(f"  After medium merge: {current_n} clusters")

        # Step 3: Aggressive merge if still too many
        if current_n > cfg.target_max_clusters:
            for _ in range(5):
                if len(get_cluster_sizes()) <= cfg.target_min_clusters:
                    break
                merge_small(cfg.merge_large_threshold, max_merges=3)

            current_n = len(get_cluster_sizes())
            print(f"  After aggressive merge: {current_n} clusters")

        return labels

    def reassign_noise(
        self,
        cluster_labels: np.ndarray,
        reduced: np.ndarray
    ) -> np.ndarray:
        """Reassign noise points to nearest clusters."""
        labels = cluster_labels.copy()
        n_noise = (labels == -1).sum()

        if n_noise > len(labels) * self.config.noise_reassign_threshold:
            print(f"\nReassigning {n_noise} noise points...")

            noise_mask = labels == -1
            noise_points = reduced[noise_mask]
            cluster_points = reduced[~noise_mask]
            cluster_ids = labels[~noise_mask]

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(cluster_points, cluster_ids)
            labels[noise_mask] = knn.predict(noise_points)

            new_noise = (labels == -1).sum()
            print(f"  After reassignment: {new_noise} noise points")

        return labels

    def select_representatives(
        self,
        df_all: pd.DataFrame,
        embeddings: np.ndarray,
        reduced: np.ndarray,
        cluster_labels: np.ndarray,
        sentences: List[str]
    ) -> pd.DataFrame:
        """Select representative sentences for each cluster."""
        print("\nSelecting representative sentences...")

        representatives = []
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                continue

            cluster_mask = df_all["cluster"] == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_sentences = np.array(sentences)[cluster_mask]
            cluster_sources = np.array(df_all["source"])[cluster_mask]
            cluster_umap = reduced[cluster_mask]

            centroid = cluster_embeddings.mean(axis=0, keepdims=True)
            closest_idx, _ = pairwise_distances_argmin_min(centroid, cluster_embeddings)

            representatives.append({
                "cluster": cluster_id,
                "size": len(cluster_sentences),
                "source": cluster_sources[closest_idx[0]],
                "representative": cluster_sentences[closest_idx[0]],
                "UMAP_1": cluster_umap[closest_idx[0], 0],
                "UMAP_2": cluster_umap[closest_idx[0], 1]
            })

        return pd.DataFrame(representatives).sort_values("size", ascending=False)

    def visualize(
        self,
        reduced: np.ndarray,
        cluster_labels: np.ndarray,
        output_path: str
    ):
        """Create UMAP visualization."""
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = (cluster_labels == -1).sum()

        plt.figure(figsize=(14, 11))
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=cluster_labels,
            cmap="tab20",
            s=12,
            alpha=0.9,
            edgecolors='k',
            linewidths=0.2
        )
        plt.title(
            f"UMAP Projection with HDBSCAN Clusters\n"
            f"{n_clusters} clusters | {n_noise} noise points",
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        plt.xlabel("UMAP 1", fontsize=12)
        plt.ylabel("UMAP 2", fontsize=12)
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(alpha=0.2, linestyle='--')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved: {output_path}")

    def run(
        self,
        input_dir: str,
        output_dir: str,
        domain: str = "default",
        text_column: str = "text"
    ) -> ClusteringResult:
        """Run the full clustering pipeline."""
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        df_all, sentences = self.load_data(input_dir, text_column)

        # Encode
        embeddings = self.encode(sentences)

        # Reduce dimensions
        reduced = self.reduce_dimensions(embeddings)
        df_all["UMAP_1"] = reduced[:, 0]
        df_all["UMAP_2"] = reduced[:, 1]

        # Cluster
        cluster_labels = self.cluster(reduced)

        # Merge small clusters
        cluster_labels = self.merge_clusters(cluster_labels, reduced)

        # Reassign noise
        cluster_labels = self.reassign_noise(cluster_labels, reduced)

        df_all["cluster"] = cluster_labels

        # Select representatives
        representatives = self.select_representatives(
            df_all, embeddings, reduced, cluster_labels, sentences
        )

        # Calculate stats
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = (cluster_labels == -1).sum()

        # Save outputs
        rep_path = os.path.join(output_dir, f"representative_sentences_{domain}.csv")
        all_path = os.path.join(output_dir, f"all_sentences_{domain}.csv")
        plot_path = os.path.join(output_dir, f"umap_clusters_{domain}.png")

        representatives.to_csv(rep_path, index=False, encoding="utf-8-sig")
        df_all.to_csv(all_path, index=False, encoding="utf-8-sig")
        self.visualize(reduced, cluster_labels, plot_path)

        # Print summary
        print(f"\n{'='*50}")
        print(f"Clustering Results for '{domain}':")
        print(f"  - Total sentences: {len(sentences):,}")
        print(f"  - Clusters: {n_clusters}")
        print(f"  - Noise points: {n_noise} ({n_noise/len(sentences)*100:.1f}%)")
        print(f"  - Avg cluster size: {representatives['size'].mean():.1f}")
        print(f"{'='*50}")

        return ClusteringResult(
            df_all=df_all,
            embeddings=embeddings,
            reduced=reduced,
            cluster_labels=cluster_labels,
            representatives=representatives,
            n_clusters=n_clusters,
            n_noise=n_noise
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Text clustering pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input directory with CSV files")
    parser.add_argument("--output", "-o", default="outputs/clustering", help="Output directory")
    parser.add_argument("--domain", "-d", default="default", help="Domain name for output files")
    parser.add_argument("--text-column", default="text", help="Text column name in CSV")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--target-clusters", type=int, default=15, help="Target cluster count")

    args = parser.parse_args()

    config = ClusteringConfig(
        embedding_model=args.model,
        device=args.device,
        target_min_clusters=args.target_clusters,
        target_max_clusters=args.target_clusters + 5,
    )

    pipeline = ClusteringPipeline(config)
    pipeline.run(args.input, args.output, args.domain, args.text_column)


if __name__ == "__main__":
    main()
