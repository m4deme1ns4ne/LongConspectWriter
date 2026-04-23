import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from src.core.base import BaseClusterVisualizer
from loguru import logger


class LocalClusterVisualizer(BaseClusterVisualizer):
    def plot(self, labels: np.ndarray, metadata: dict = None):
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

        self._save_and_close("02_local_clusters", "local_distribution", metadata)


class GlobalClusterVisualizer(BaseClusterVisualizer):
    def plot(
        self,
        embeddings: np.ndarray,
        assignments: list,
        chapter_titles: list,
        metadata: dict = None,
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

        self._save_and_close("05_global_clusters", "global_distribution", metadata)
