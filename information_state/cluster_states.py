from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .utils import (
    read_dataframe,
    resolve_existing_table,
    write_dataframe,
    write_json,
    make_project_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster latent state embeddings into candidate phenotypes.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--embeddings-path", type=str, default=None)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--k", nargs="+", type=int, default=[4])
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_project_config(project_root=args.project_root)
    embeddings_path = Path(args.embeddings_path).resolve() if args.embeddings_path else config.embeddings_dir / f"{args.split}_embeddings.npy"
    metadata_path = Path(args.metadata_path).resolve() if args.metadata_path else resolve_existing_table(
        config.embeddings_dir / f"{args.split}_metadata.parquet"
    )

    embeddings = np.load(embeddings_path)
    metadata = read_dataframe(metadata_path).reset_index(drop=True)
    if len(metadata) != len(embeddings):
        raise ValueError("Embedding rows and metadata rows do not align.")
    if len(embeddings) == 0:
        raise ValueError("No embeddings were found for clustering.")

    config.clusters_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    summary = {"split": args.split, "runs": []}
    best = None

    for k in args.k:
        if k <= 1:
            raise ValueError("k must be greater than 1.")
        if len(embeddings) < k:
            raise ValueError(f"k={k} exceeds available windows ({len(embeddings)}).")

        model = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = model.fit_predict(scaled_embeddings)
        silhouette = float(silhouette_score(scaled_embeddings, labels)) if len(np.unique(labels)) > 1 else float("nan")
        davies_bouldin = float(davies_bouldin_score(scaled_embeddings, labels)) if len(np.unique(labels)) > 1 else float("nan")

        labels_path = config.clusters_dir / f"kmeans_k{k}_labels.npy"
        np.save(labels_path, labels)
        np.savez(
            config.clusters_dir / f"kmeans_k{k}_model.npz",
            centers=model.cluster_centers_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            k=np.array([k], dtype=np.int32),
        )

        assignments = metadata.copy()
        assignments["cluster"] = labels.astype(int)
        assignments["cluster_method"] = "kmeans"
        assignments["k"] = int(k)
        assignments_path = write_dataframe(assignments, config.clusters_dir / f"kmeans_k{k}_assignments.parquet")

        run_summary = {
            "k": int(k),
            "num_windows": int(len(assignments)),
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "cluster_sizes": assignments["cluster"].value_counts().sort_index().astype(int).to_dict(),
            "assignments_path": str(assignments_path),
            "labels_path": str(labels_path),
            "model_path": str(config.clusters_dir / f"kmeans_k{k}_model.npz"),
        }
        summary["runs"].append(run_summary)

        if best is None or (np.isnan(best["silhouette"]) and not np.isnan(silhouette)) or silhouette > best["silhouette"]:
            best = {
                "k": int(k),
                "silhouette": silhouette,
                "assignments": assignments,
                "model": model,
            }

    if best is None:
        raise RuntimeError("Failed to compute any clustering run.")

    selected_assignments_path = write_dataframe(best["assignments"], config.clusters_dir / "cluster_assignments.parquet")
    np.savez(
        config.clusters_dir / "cluster_model.npz",
        centers=best["model"].cluster_centers_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        k=np.array([best["k"]], dtype=np.int32),
    )

    summary["selected_k"] = int(best["k"])
    summary["selected_assignments_path"] = str(selected_assignments_path)
    summary["selected_model_path"] = str(config.clusters_dir / "cluster_model.npz")
    write_json(summary, config.clusters_dir / "cluster_summary.json")


if __name__ == "__main__":
    main()
