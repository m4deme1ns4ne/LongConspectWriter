import json
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

import matplotlib

matplotlib.use("Agg")

from src.core.utils import TextsSplitter
from src.core.base import BaseLocalClusterizer, BaseGlobalClusterizer
from src.core.vizualization import LocalClusterVisualizer, GlobalClusterVisualizer


class SemanticLocalClusterizer(BaseLocalClusterizer):
    def __init__(self, init_config, gen_config, session_dir: Path):
        self.session_dir = session_dir
        super().__init__(init_config, gen_config)
        self.model = SentenceTransformer(self._init_config.model_name)

    def run(self, path):
        with open(path, "r", encoding="utf-8") as file:
            transcrib = json.load(file)

        sentences = TextsSplitter.split_text_to_sentences(transcrib["answer_agent"])
        logger.info(f"Всего предложений: {len(sentences)}")

        embeddings = self.model.encode(sentences)
        n_samples = len(embeddings)

        if self._gen_config.turn_on_connectivity is True:
            connectivity = np.eye(n_samples, k=1) + np.eye(n_samples, k=-1)
        else:
            connectivity = None

        local_clusterer = AgglomerativeClustering(
            n_clusters=self._gen_config.n_clusters,
            distance_threshold=self._gen_config.threshold,
            metric=self._gen_config.metric,
            linkage=self._gen_config.linkage,
            connectivity=connectivity,
        )

        raw_labels = local_clusterer.fit_predict(embeddings)

        min_s = 5
        clusters, final_labels = self._format_cluster_output(
            sentences, raw_labels, min_sentences=min_s
        )

        metadata = {
            "Model": self._init_config.model_name,
            "Threshold": self._gen_config.threshold,
            "Linkage": self._gen_config.linkage,
            "Sentences": len(sentences),
            "Min_Sentences_Buffer": min_s,
        }

        visualizer = LocalClusterVisualizer(self.session_dir)
        visualizer.run(final_labels, metadata)

        out_filepath = self._safe_result_out_line(
            output=clusters,
            stage="02_local_clusters/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.info(f"Финальных локальных кластеров: {len(clusters)}")

        return out_filepath

    def _format_cluster_output(self, sentences, raw_labels, min_sentences=5):
        chunks = []
        current_chunk = []
        current_label = raw_labels[0]

        for sentence, label in zip(sentences, raw_labels):
            if label != current_label and len(current_chunk) >= min_sentences:
                chunks.append(current_chunk)
                current_chunk = []

            current_label = label
            current_chunk.append(sentence)

        if current_chunk:
            if len(current_chunk) < min_sentences and chunks:
                chunks[-1].extend(current_chunk)
            else:
                chunks.append(current_chunk)

        formated_clusters = {}
        final_labels = []

        for i, chunk in enumerate(chunks):
            formated_clusters[i] = " ".join(chunk)
            final_labels.extend([i] * len(chunk))

        return formated_clusters, np.array(final_labels)


class SemanticGlobalClusterizer(BaseGlobalClusterizer):
    def __init__(self, init_config, session_dir: Path):
        self.session_dir = session_dir
        super().__init__(init_config)
        self.model = SentenceTransformer(self._init_config.model_name)

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
        _, assignments_tensor = torch.max(scores, dim=1)
        assignments = assignments_tensor.tolist()

        smoothed_assignments = []
        current_chapter = 0  # Начинаем с первой главы

        for i, chapter in enumerate(assignments):
            # Если модель предлагает прыгнуть назад во времени — игнорируем
            if chapter < current_chapter:
                smoothed_assignments.append(current_chapter)

            # Если модель предлагает прыгнуть вперед
            elif chapter > current_chapter:
                # Проверяем, не случайный ли это выброс (смотрим на след. 2 куска)
                future_window = assignments[i : i + 3]

                # Если в будущем глава стабильно новая, переходим в нее
                # Иначе считаем это галлюцинацией e5 и остаемся в текущей
                if future_window.count(chapter) >= 2 or i == len(assignments) - 1:
                    current_chapter = chapter

                smoothed_assignments.append(current_chapter)
            else:
                smoothed_assignments.append(current_chapter)

        # Перезаписываем assignments сглаженной версией
        assignments = smoothed_assignments

        metadata = {
            "Model": self._init_config.model_name,
            "Metric": "cosine",
            "Chapters": len(chapter_titles),
        }
        visualizer = GlobalClusterVisualizer(self.session_dir)
        visualizer.run(local_clusters_embeddings, assignments, chapter_titles, metadata)

        for chunk_idx, chapter_idx in enumerate(assignments):
            global_clusters[chapter_titles[chapter_idx]].append(
                local_clusters[chunk_idx]
            )

        global_clusters = {
            key: value for key, value in global_clusters.items() if value
        }

        out_filepath = self._safe_result_out_line(
            output=global_clusters,
            stage="05_global_clusters/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.info(f"Глобальных кластеров: {len(global_clusters)}")
        logger.debug(f"Type local clusters: {type(global_clusters)}")

        return out_filepath
