"""Этапы семантической кластеризации для пайплайна LongConspectWriter.

Модуль преобразует транскрипт в хронологические локальные кластеры, а затем
сопоставляет их с главами из глобального плана. Контракт пайплайна сохраняется:
результат каждого этапа сохраняется, а наружу возвращается путь к артефакту.
"""

import json
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import sentence_transformers
from sklearn.cluster import AgglomerativeClustering
from src.configs.configs import (
    GlobalClusterizerInitConfig,
    LocalClusterizerGenConfig,
    LocalClusterizerInitConfig,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Any
from src.core.utils import TextsSplitter
from src.core.base import (
    BaseLocalClusterizer,
    BaseGlobalClusterizer,
    BaseClusterVisualizer,
)


class SemanticLocalClusterizer(BaseLocalClusterizer):
    """Строит хронологические семантические кластеры из сырого транскрипта."""

    def __init__(
        self,
        init_config: LocalClusterizerInitConfig,
        gen_config: LocalClusterizerGenConfig,
        session_dir: Path,
    ) -> None:
        """Инициализирует модель sentence embeddings для локального этапа.

        Args:
            init_config (LocalClusterizerInitConfig): Настройки embedding-модели
                для локальной кластеризации.
            gen_config (LocalClusterizerGenConfig): Параметры агломеративной кластеризации
                для хронологических чанков транскрипта.
            session_dir (Path): Директория текущей сессии пайплайна.

        Returns:
            None: Кластеризатор сохраняет конфиг и загруженную embedding-модель.
        """
        self.session_dir = session_dir
        super().__init__(init_config, gen_config)
        self.model = sentence_transformers.SentenceTransformer(
            self._init_config.model_name, device=self._init_config.device
        )

    def run(self, path: str | Path) -> Path:
        """Кластеризует предложения транскрипта и сохраняет JSON локальных кластеров.

        Args:
            path (str | Path): Путь к STT-артефакту с текстом
                ``answer_agent``.

        Returns:
            Path: Путь к сохраненному артефакту локальных кластеров для планирования.
        """
        with open(path, "r", encoding="utf-8") as file:
            transcrib = json.load(file)

        sentences = TextsSplitter.split_text_to_sentences(transcrib["answer_agent"])
        logger.debug(f"Всего предложений: {len(sentences)}")

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

    def _format_cluster_output(
        self,
        sentences: list[str],
        raw_labels: np.ndarray,
        min_sentences: int = 5,
    ) -> tuple[dict[int, str], np.ndarray]:
        """Преобразует сырые метки кластеризации в упорядоченные текстовые кластеры.

        Args:
            sentences (list[str]): Предложения транскрипта в хронологическом порядке лекции.
            raw_labels (np.ndarray): Сырые метки, возвращенные агломеративной кластеризацией.
            min_sentences (int): Минимальное число предложений до закрытия чанка
                в отдельный локальный кластер.

        Returns:
            tuple[dict[int, str], np.ndarray]: Текст кластеров с ключом по индексу кластера
            и финальная последовательность меток для визуализации.
        """
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
    """Назначает локальные кластеры заголовкам глав из глобального плана."""

    def __init__(
        self, init_config: GlobalClusterizerInitConfig, session_dir: Path
    ) -> None:
        """Инициализирует embedding-модель для глобального распределения кластеров.

        Args:
            init_config (GlobalClusterizerInitConfig): Настройки embedding-модели
                для выравнивания с глобальными главами.
            session_dir (Path): Директория текущей сессии пайплайна.

        Returns:
            None: Кластеризатор сохраняет конфиг и загруженную embedding-модель.
        """
        self.session_dir = session_dir
        super().__init__(init_config)
        self.model = sentence_transformers.SentenceTransformer(
            self._init_config.model_name, device=self._init_config.device
        )

    def run(self, plan_path: str | Path, local_clusters_path: str | Path) -> Path:
        """Сопоставляет локальные кластеры с глобальными кластерами уровня глав.

        Args:
            plan_path (str | Path): Путь к JSON глобального плана с записями
                ``chapters``.
            local_clusters_path (str | Path): Путь к JSON локальных кластеров из
                ``SemanticLocalClusterizer``.

        Returns:
            Path: Путь к сохраненному артефакту глобальных кластеров для синтеза.
        """

        logger.debug("Открываю global_plan JSON...")
        with open(plan_path, "r", encoding="utf-8") as file:
            global_plan = json.load(file)

        logger.debug("Открываю local_clusters JSON...")
        with open(local_clusters_path, "r", encoding="utf-8") as file:
            local_clusters_dict = json.load(file)

        local_clusters = list(local_clusters_dict.values())

        chapters = global_plan["chapters"]
        chapter_titles = [ch["chapter_title"] for ch in chapters]
        logger.debug(
            f"Глав: {len(chapters)}, локальных кластеров: {len(local_clusters)}"
        )

        logger.debug("Кодирую главы (global_plan_embeddings)...")
        global_plan_embeddings = self.model.encode(
            [
                f"query: {dict_chapter['chapter_title']}. {dict_chapter['description']}"
                for dict_chapter in chapters
            ]
        )
        logger.debug(f"global_plan_embeddings: {global_plan_embeddings.shape}")

        logger.debug("Кодирую локальные кластеры (local_clusters_embeddings)...")
        local_clusters_embeddings = self.model.encode(
            [f"passage: {clusters}" for clusters in local_clusters]
        )
        logger.debug(f"local_clusters_embeddings: {local_clusters_embeddings.shape}")

        logger.debug("Вычисляю cos_sim и torch.max...")
        global_clusters = {key: [] for key in chapter_titles}
        scores = sentence_transformers.util.cos_sim(
            local_clusters_embeddings, global_plan_embeddings
        )
        _, assignments_tensor = torch.max(scores, dim=1)
        assignments = assignments_tensor.tolist()
        logger.debug(f"Назначения: {len(assignments)} кластеров распределено")

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
        logger.debug("Запускаю GlobalClusterVisualizer...")
        visualizer = GlobalClusterVisualizer(self.session_dir)
        visualizer.run(local_clusters_embeddings, assignments, chapter_titles, metadata)
        logger.debug("GlobalClusterVisualizer завершён.")

        for chunk_idx, chapter_idx in enumerate(assignments):
            global_clusters[chapter_titles[chapter_idx]].append(
                local_clusters[chunk_idx]
            )

        global_clusters = {
            key: value for key, value in global_clusters.items() if value
        }

        logger.debug(
            f"Сохраняю результат, глобальных кластеров: {len(global_clusters)}"
        )
        out_filepath = self._safe_result_out_line(
            output=global_clusters,
            stage="05_global_clusters/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.info(f"Глобальных кластеров: {len(global_clusters)}")
        logger.debug(f"Type local clusters: {type(global_clusters)}")

        return out_filepath


class LocalClusterVisualizer(BaseClusterVisualizer):
    """Отрисовывает хронологическое распределение локальных кластеров."""

    def run(self, labels: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
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
    """Отрисовывает семантическую проекцию локальных кластеров по глобальным главам."""

    def run(
        self,
        embeddings: np.ndarray,
        assignments: list[int],
        chapter_titles: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(embeddings) < 2:
            logger.warning("Слишком мало данных для PCA проекции.")
            return

        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        unique_chapters = list(dict.fromkeys(assignments))
        colors = plt.cm.tab10.colors
        color_map = {
            ch: colors[i % len(colors)] for i, ch in enumerate(unique_chapters)
        }

        plt.figure(figsize=(14, 8))
        for ch_idx in unique_chapters:
            mask = [i for i, a in enumerate(assignments) if a == ch_idx]
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                label=f"Ch {ch_idx + 1}",
                color=color_map[ch_idx],
                s=100,
                alpha=0.8,
            )

        plt.title("Global cluster distribution (PCA)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        self._save_and_close("05_global_clusters", "global_distribution", metadata)
