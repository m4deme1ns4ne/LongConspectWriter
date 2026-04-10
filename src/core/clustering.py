from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from pprint import pprint
from src.core.utils import TextsSplitter
from loguru import logger
import numpy as np
import umap
import hdbscan


class SemanticLocalClusterizer:
    def __init__(self, model):
        self.model = SentenceTransformer(model)

    def build_local_clusters(self, transcrib):
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

        return clusters

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

        return new_grouped_data
    

class SemanticGlobalClusterizer:
    def __init__(self, model):
        self.model = SentenceTransformer(model)

    def build_global_clusters(self, local_clusters):

        formatted_texts = [f"passage: {text}" for text in local_clusters]
        embeddings = self.model.encode(formatted_texts, show_progress_bar=True, normalize_embeddings=False)

        normalized_embeddings = normalize(embeddings, norm="l2")

        umap_model = umap.UMAP(
            n_neighbors=min(15, len(formatted_texts) - 1), 
            n_components=10, # Сжимаем 768 -> 10 измерений
            metric="euclidean",
            random_state=42 # Важно для воспроизводимости пайплайна
        )

        reduced_embeddings = umap_model.fit_transform(normalized_embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=10,
            metric="euclidean",
            cluster_selection_epsilon=0.1, 
            cluster_selection_method="eom" # "eom" (Excess of Mass) лучше для поиска крупных тем
        )

        labels = clusterer.fit_predict(reduced_embeddings)

        # Извлечение результатов
        topics = {}
        noise_count = 0

        for idx, label in enumerate(labels):
            if label == -1:
                noise_count += 1
                continue # Игнорируем семантический мусор лектора
                
            if label not in topics:
                topics[label] = []
                
            topics[label].append(local_clusters[idx])

        topics = [value for keys, value in topics.items()]

        return topics



with open(
    "data/example-transcrib/large-v3-turbo-cuda-float16-Лекция 1. -1775312903.txt",
    "r",
    encoding="utf-8",
) as file:
    transcrib = file.read()


model_local_clustering = SemanticLocalClusterizer("cointegrated/rubert-tiny2")
local_clusters = model_local_clustering.build_local_clusters(transcrib)

logger.success(f"Локальных кластеров: {len(local_clusters)}")
logger.debug(f"Type local clusters: {type(local_clusters)}")

local_clusters = [" ".join(_["text"]) for _ in local_clusters]

model_global_clustering = SemanticGlobalClusterizer("intfloat/multilingual-e5-base")
global_clusters = model_global_clustering.build_global_clusters(local_clusters)

logger.success(f"Глобальных кластеров: {len(global_clusters)}")
logger.debug(f"Type global clusters: {type(global_clusters)}")

pprint(
    global_clusters
)
