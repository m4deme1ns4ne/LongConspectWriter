from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from src.core.utils import TextsSplitter
from loguru import logger
import numpy as np
import time
import os
from pathlib import Path
from src.core.utils import SEPARATOR
import json
from sentence_transformers import util
import torch
from src.agents.base_agent import Trackable


class BaseClusterizer(Trackable):
    pass


class SemanticLocalClusterizer(BaseClusterizer):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def run(self, path):
        with open(path, "r", encoding="utf-8") as file:
            transcrib = file.read()

        sentences = TextsSplitter.split_text_to_sentences(transcrib)
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

        monolith_draft = f"\n\n{SEPARATOR}\n\n".join(clusters)

        safe_model_name = self.model_name.replace("/", "_")

        pure_transcrib_file_name = os.path.basename(path)

        timestamp = int(time.time())
        out_filepath = os.path.join(
            Path("data/example-clusters/example-local-clusters"),
            f"{safe_model_name}-{pure_transcrib_file_name}-{timestamp}.txt",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            f.write(monolith_draft)

        logger.success(f"Локальные кластеры сохранены в {out_filepath}")
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

        return new_grouped_data


class SemanticGlobalClusterizer(BaseClusterizer):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def run(self, plan_path, local_clusters_path):

        with open(plan_path, "r", encoding="utf-8") as file:
            global_plan = json.load(file)

        with open(local_clusters_path, "r", encoding="utf-8") as file:
            local_clusters = file.read()

        local_clusters = [
            chunk.strip() for chunk in local_clusters.split(SEPARATOR) if chunk.strip()
        ]

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

        safe_model_name = self.model_name.replace("/", "_")

        timestamp = int(time.time())
        out_filepath = os.path.join(
            Path("data/example-clusters/example-global-clusters"),
            f"{safe_model_name}-{timestamp}.json",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as file:
            json.dump(global_clusters, file, ensure_ascii=False, indent=4)

        logger.info(f"Глобальных кластеров: {len(global_clusters)}")
        logger.debug(f"Type local clusters: {type(global_clusters)}")
        logger.success(f"Глобальные кластеры сохранены в {out_filepath}")

        return out_filepath
