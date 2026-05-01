"""Агент финального синтеза для пайплайна LongConspectWriter.

Агент пишет длинный конспект тема за темой из глобальных кластеров и использует
extractor-помощник, чтобы сохранить существующую семантику словаря контекста
для уже разобранных сущностей лекции.
"""

from tqdm import tqdm
import sys
from pathlib import Path
import json

from loguru import logger
from src.core.base import BaseLlamaCppAgent
from src.core.utils import TextsSplitter, ColoursForTqdm
from src.agents.agent_extractor import _AgentExtractor
from src.configs.configs import LLMAppConfig, LLMGenConfig
from typing import Any


class AgentSynthesizerLlama(BaseLlamaCppAgent):
    """Синтезирует итоговый JSON-конспект из глобальных кластеров."""

    def __init__(
        self,
        session_dir: Path,
        extractor_gen_config: LLMGenConfig,
        extractor_app_config: LLMAppConfig,
        **kwargs: Any,
    ) -> None:
        """Инициализирует конфигурацию синтезатора и extractor.

        Args:
            session_dir (Path): Директория текущей сессии пайплайна.
            extractor_gen_config (LLMGenConfig): Generation settings for the
                per-chunk extractor helper.
            extractor_app_config (LLMAppConfig): App settings and schema paths
                для extractor-помощника.
            **kwargs (Any): LLM configuration passed to ``BaseLlamaCppAgent``.

        Returns:
            None: Синтезатор сохраняет сессию, модель и конфиг extractor.

        Raises:
            Exception: Пробрасывает ошибки инициализации базового агента.
        """
        self.session_dir = session_dir
        self.extractor_gen_config = extractor_gen_config
        self.extractor_app_config = extractor_app_config

        self.lecture_theme = kwargs.get("lecture_theme", "universal")

        super().__init__(**kwargs)

    def _generate_synthesizer_chunk(
        self,
        chunk: str,
        topik: str,
        already_seen_themes: set[str],
        last_tail: str,
    ) -> str:
        """Генерирует текст конспекта для одного чанка темы.

        Args:
            chunk (str): Текстовый чанк из текущего глобального кластера.
            topik (str): Заголовок текущей глобальной темы.
            already_seen_themes (set[str]): Entities already extracted from
                previous synthesized chunks.
            last_tail (str): Хвост предыдущего сгенерированного чанка для связности.

        Returns:
            str: Синтезированный текст конспекта для чанка.

        Raises:
            Exception: Пробрасывает ошибки генерации LLM.
        """
        with tqdm(
            total=self._gen_config.max_tokens,
            desc="Генерация токенов для чанка",
            unit="токен",
            colour=ColoursForTqdm.fourth_level,
            leave=False,
            position=2,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as chunk_token_pbar:
            seen_str = (
                ", ".join(already_seen_themes)
                if already_seen_themes
                else "Ничего ещё не было разобрано."
            )

            combined_context = (
                f"[УЖЕ РАЗОБРАНО]: {seen_str}\n[КОНЕЦ ПРЕДЫДУЩЕГО ЧАНКА]: {last_tail}"
            )

            response = self._generate(
                prompt=self._build_prompt(
                    chunk=chunk,
                    cluster_topik=topik,
                    previous_context=combined_context,
                    tokenizer=self.model.tokenize,
                ),
                stream=True,
                token_pbar=chunk_token_pbar,
            )
        return response

    def _generate_synthesizer(
        self,
        split_clusters: list[str],
        topik: str,
        conspect: dict[str, list[str]],
        already_seen_themes: set[str],
    ) -> None:
        """Генерирует все чанки для одной глобальной темы и обновляет контекст.

        Args:
            split_clusters (list[str]): Токен-размерные чанки текста темы.
            topik (str): Заголовок текущей глобальной темы.
            conspect (dict[str, list[str]]): Mutable conspect accumulator keyed
                by topic title.
            already_seen_themes (set[str]): Mutable context of extracted terms
                shared across synthesized chunks.

        Returns:
            None: Аккумуляторы конспекта и контекста обновляются на месте.

        Raises:
            Exception: Пробрасывает ошибки синтезатора или extractor.
        """
        extractor = _AgentExtractor(
            init_config=self._init_config,
            gen_config=self.extractor_gen_config,
            app_config=self.extractor_app_config,
            session_dir=self.session_dir,
            shared_model=self.model,
            lecture_theme=self.lecture_theme,
        )
        last_tail = "[НАЧАЛО ДОКУМЕНТА, ПРОДОЛЖАЙ ТЕКСТ]"
        with tqdm(
            total=len(split_clusters),
            desc="Чанки",
            unit="чанк",
            colour=ColoursForTqdm.second_level,
            leave=False,
            position=1,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as chunk_pbar:
            for chunk in split_clusters:
                synthesize_chunk = self._generate_synthesizer_chunk(
                    chunk=chunk,
                    topik=topik,
                    already_seen_themes=already_seen_themes,
                    last_tail=last_tail,
                )
                conspect[topik].append(synthesize_chunk)
                last_tail = " ".join(
                    synthesize_chunk.split()[-self._app_config.last_tail_words_count :]
                )
                extracted_dict = extractor.run(synthesizer_chunk=synthesize_chunk)
                for term in extracted_dict.get("extracted_entities", []):
                    already_seen_themes.add(term.lower())
                chunk_pbar.update(1)

    def run(self, path: str | Path) -> Path:
        """Синтезирует итоговый JSON-конспект из артефакта глобальных кластеров.

        Args:
            path (str | Path): Путь к глобальным кластерам, сгруппированным по главам/темам.

        Returns:
            Path: Путь к JSON-артефакту синтезированного конспекта.

        Raises:
            OSError: Если нет доступа к входным или выходным артефактам.
            json.JSONDecodeError: Если глобальные кластеры не являются валидным JSON.
            Exception: Пробрасывает ошибки разбиения, генерации или извлечения.
        """
        with open(path, "r", encoding="utf-8") as file:
            global_clusters = json.load(file)

        conspect = {topik: [] for topik, _ in global_clusters.items()}
        already_seen_themes = set()

        with tqdm(
            total=len(global_clusters),
            unit="кластер",
            desc="Темы",
            colour=ColoursForTqdm.first_level,
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for topik, cluster_text in global_clusters.items():
                full_text = " ".join(cluster_text)
                split_clusters: list[str] = TextsSplitter.split_text_to_tokens(
                    text=full_text,
                    tokenizer=self.model.tokenize,
                    chunk_size=int(
                        self._app_config.chunk_size_ratio * self._gen_config.max_tokens
                    ),
                    chunk_overlap=int(
                        self._app_config.chunk_overlap_ratio
                        * self._gen_config.max_tokens
                    ),
                )

                self._generate_synthesizer(
                    topik=topik,
                    conspect=conspect,
                    split_clusters=split_clusters,
                    already_seen_themes=already_seen_themes,
                )

                pbar.update(1)

        logger.info("Генерация финального конспекта завершена.")

        out_filepath = self._safe_result_out_line(
            output=conspect,
            stage=self._app_config.name_stage_dir,
            file_name="conspect.json",
            session_dir=self.session_dir,
        )

        logger.success(f"Финальный конспект сохранен: {out_filepath}")
        return out_filepath
