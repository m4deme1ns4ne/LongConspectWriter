import json
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.utils import TextsSplitter
from src.core.base import BaseClusterizer


class Visualize_clustering_metrics:
    @staticmethod
    def plot_local_clusters(labels: np.ndarray, session_dir: Path):
        """Отрисовка хронологической гистограммы локальных кластеров."""
        cluster_sizes = {}
        for label in labels:
            label = int(label)
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        plt.figure(figsize=(12, 6))
        plt.bar(
            range(len(cluster_sizes)),
            list(cluster_sizes.values()),
            color="skyblue",
            edgecolor="black",
        )
        plt.title("Распределение предложений по локальным кластерам (Хронология)")
        plt.xlabel("Индекс локального кластера (время лекции ->)")
        plt.ylabel("Количество предложений в кластере")

        out_path = session_dir / "02_local_clusters" / "local_distribution.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.success(f"График локальной кластеризации сохранен: {out_path}")

    @staticmethod
    def plot_global_clusters(
        embeddings: np.ndarray,
        assignments: list,
        chapter_titles: list,
        session_dir: Path,
    ):
        """Отрисовка 2D PCA проекции привязки абзацев к главам."""
        if len(embeddings) < 2:
            logger.warning("Слишком мало данных для PCA проекции.")
            return

        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        assigned_labels = [chapter_titles[idx] for idx in assignments]

        plt.figure(figsize=(14, 8))
        sns.scatterplot(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            hue=assigned_labels,
            palette="tab10",
            s=100,
            alpha=0.8,
        )

        plt.title("Семантическое распределение локальных кластеров по главам (PCA)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Главы")
        plt.tight_layout()

        out_path = session_dir / "05_global_clusters" / "global_distribution.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.success(f"График глобальной кластеризации сохранен: {out_path}")


class SemanticLocalClusterizer(BaseClusterizer):
    def __init__(self, model_name, session_dir: Path):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.session_dir = session_dir

    def run(self, path):
        with open(path, "r", encoding="utf-8") as file:
            transcrib = json.load(file)

        sentences = TextsSplitter.split_text_to_sentences(transcrib["answer_agent"])
        logger.info(f"Всего предложений: {len(sentences)}")

        embeddings = self.model.encode(sentences)
        n_samples = len(embeddings)

        connectivity = np.eye(n_samples, k=1) + np.eye(n_samples, k=-1)
        local_clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            metric="cosine",
            linkage="average",
            connectivity=connectivity,
        )
        labels = local_clusterer.fit_predict(embeddings)

        Visualize_clustering_metrics.plot_local_clusters(labels, self.session_dir)

        clusters = self._format_cluster_output(sentences, labels)

        out_filepath = self._safe_result_out_line(
            output_dict=clusters,
            stage="02_local_clusters/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.info(f"Локальных кластеров: {len(clusters)}")
        logger.debug(f"Type local clusters: {type(clusters)}")

        return out_filepath

    def _format_cluster_output(self, sentences, labels):
        grouped_data = {}
        for index, (items, label) in enumerate(zip(sentences, labels)):
            label = int(label)

            if label not in grouped_data:
                grouped_data[label] = {"index": index, "text": []}

            grouped_data[label]["text"].append(items)

        grouped_data = sorted(grouped_data.items(), key=lambda index: index[1]["index"])
        new_grouped_data = []
        for cluster in grouped_data:
            new_grouped_data.append(cluster[1])

        new_grouped_data = [" ".join(_["text"]) for _ in new_grouped_data]

        formated_clusters = {}
        for i, cluster in enumerate(new_grouped_data):
            formated_clusters[i] = cluster

        return formated_clusters


class SemanticGlobalClusterizer(BaseClusterizer):
    def __init__(self, model_name, session_dir: Path):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.session_dir = session_dir

    def run(self, plan_path, local_clusters_path):

        with open(plan_path, "r", encoding="utf-8") as file:
            global_plan = json.load(file)

        with open(local_clusters_path, "r", encoding="utf-8") as file:
            local_clusters_dict = json.load(file)

        local_clusters = list(local_clusters_dict.values())

        chapters = global_plan["chapters"]
        chapter_titles = [ch["chapter_title"] for ch in chapters]

        global_plan_embeddings = self.model.encode(
            [
                f"query: {dict_chapter['chapter_title']}. {dict_chapter['description']}"
                for dict_chapter in chapters
            ]
        )
        local_clusters_embeddings = self.model.encode(
            [f"passage: {clusters}" for clusters in local_clusters]
        )

        global_clusters = {key: [] for key in chapter_titles}
        scores = util.cos_sim(local_clusters_embeddings, global_plan_embeddings)

        max_scores_tensor, assignments_tensor = torch.max(scores, dim=1)
        max_scores_tensor, assignments_tensor = (
            max_scores_tensor.tolist(),
            assignments_tensor.tolist(),
        )

        Visualize_clustering_metrics.plot_global_clusters(
            local_clusters_embeddings,
            assignments_tensor,
            chapter_titles,
            self.session_dir,
        )

        for chunk_idx, chapter_idx in enumerate(assignments_tensor):
            global_clusters[chapter_titles[chapter_idx]].append(
                local_clusters[chunk_idx]
            )

        global_clusters = {
            key: value for key, value in global_clusters.items() if value
        }

        out_filepath = self._safe_result_out_line(
            output_dict=global_clusters,
            stage="05_global_clusters/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.info(f"Глобальных кластеров: {len(global_clusters)}")
        logger.debug(f"Type local clusters: {type(global_clusters)}")

        return out_filepath
