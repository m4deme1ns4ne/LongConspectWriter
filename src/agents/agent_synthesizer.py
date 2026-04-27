from tqdm import tqdm
import sys
from pathlib import Path
import json

from loguru import logger
from src.core.base import BaseLlamaCppAgent
from src.core.utils import TextsSplitter, ColoursForTqdm
from src.agents.agent_extractor import _AgentExtractor


class AgentSynthesizerLlama(BaseLlamaCppAgent):
    def __init__(
        self, session_dir: Path, extractor_gen_config, extractor_app_config, **kwargs
    ):
        self.session_dir = session_dir
        self.extractor_gen_config = extractor_gen_config
        self.extractor_app_config = extractor_app_config

        self.lecture_theme = kwargs.get("lecture_theme", "universal")

        super().__init__(**kwargs)

    def _generate_synthesizer_chunk(self, chunk, topik, already_seen_themes, last_tail):
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
        self, split_clusters, topik, conspect, already_seen_themes
    ):
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

    def run(self, path: Path) -> str:
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
