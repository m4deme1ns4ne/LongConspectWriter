from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from src.core.utils import TextsSplitter
from loguru import logger
import numpy as np
from pathlib import Path
import json
from sentence_transformers import util
import torch
from src.core.base import BaseClusterizer


class Visualize_clustering_metrics:
    # TODO: [ТЕХДОЛГ - Observability] Реализовать визуализацию распределения кластеров
    # Использовать: plotly (для интерактива) или matplotlib/seaborn (для статики) + umap-learn для проекций
    def _visualize_clustering_metrics(
        self,
        sentences: list,
        local_labels: list,
        global_clusters: dict,
        local_embeddings: np.ndarray,
    ):
        """
        Отрисовка метрик кластеризации для калибровки порогов (distance_threshold) и отлова галлюцинаций матчинга.

        # 1. Локальная кластеризация (Хронологическая гистограмма)
        # Данные: local_labels -> подсчет частоты (сколько sentences в каждом label).
        # График: Bar Chart.
        # Ось X: Индекс кластера (строго по порядку времени 0, 1, 2...).
        # Ось Y: Количество предложений (размер кластера).
        # Ожидание: Отсутствие "пилы" (чередования кластеров по 1 предложению) и гигантских монолитов.

        # 2. Глобальная кластеризация (Семантический Scatter Plot)
        # Данные: local_embeddings (векторы e5-small) -> сжать до 2D через PCA или UMAP.
        # График: 2D Scatter Plot.
        # Точки: Локальные кластеры.
        # Цвет (hue): Название главы из global_clusters, к которой привязан кластер.
        # Ожидание: Четкие визуальные облака точек. Если точка далеко от своего облака — это мусор,
        # который алгоритм притянул к главе "за уши". Нужно вводить threshold для косинусного расстояния.
        """
        pass


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
