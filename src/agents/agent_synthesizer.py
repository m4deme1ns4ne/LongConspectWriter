import os
from tqdm import tqdm
import time
import sys
from pathlib import Path
import json

from loguru import logger
from src.core.base import BaseLlamaCppAgent
from src.core.utils import TextsSplitter, bad_words_id_generate, ColoursForTqdm

from src.configs.bad_words import BAD_WORDS_SYNTHESIZER


class AgentSynthesizerLlama(BaseLlamaCppAgent):
    def __init__(self, init_config, gen_config, app_config, session_dir: Path):
        self.session_dir = session_dir
        super().__init__(init_config, gen_config, app_config)

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            global_clusters = json.load(file)

        conspect = {topik: [] for topik, _ in global_clusters.items()}

        max_tokens_for_summary = int(self._gen_config.max_tokens * 0.2)
        rolling_summary = []
        last_tail = "Это начало лекции."
        max_history_tokens = 1024

        # УРОВЕНЬ 1: Цикл по глобальным темам (Кластерам)
        with tqdm(
            total=len(global_clusters),
            unit="кластер",
            desc="Темы",
            colour=ColoursForTqdm.first_level,
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for topik, clusters in global_clusters.items():
                full_text = " ".join(clusters)
                split_clusters: list[str] = TextsSplitter.split_text_to_tokens(
                    text=full_text,
                    tokenizer=self.model.tokenize,
                    chunk_size=int(self._gen_config.max_tokens * 0.75),
                    chunk_overlap=int(self._gen_config.max_tokens * 0.25),
                )

                # УРОВЕНЬ 2: Цикл по кускам текста внутри темы
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
                        # УРОВЕНЬ 3: Цикл генерации токенов для одного куска
                        history = "\n".join(rolling_summary)
                        history_tokens = len(
                            self.model.tokenize(history.encode("utf-8"))
                        )
                        if history_tokens > max_history_tokens:
                            logger.warning(
                                f"Контекст превысил максимальное кол-во токенов: {max_history_tokens}. И составляет {history_tokens} токенов."
                            )
                            logger.warning("Запуск компрессора...")
                            mega_prompt = [
                                {
                                    "role": "system",
                                    "content": self.prompts[
                                        self._app_config.agent_name
                                    ]["mega_compressor"],
                                },
                                {
                                    "role": "user",
                                    "content": f"Скомпрессируй эти тезисы:\n{history}",
                                },
                            ]
                            original_max_tokens = self._gen_config.max_tokens
                            self._gen_config.max_tokens = int(
                                self._gen_config.max_tokens * 0.3
                            )

                            with tqdm(
                                total=self._gen_config.max_tokens,
                                desc="Сжатие истории",
                                unit="токен",
                                colour=ColoursForTqdm.third_level,
                                leave=False,
                                position=2,
                                file=sys.stdout,
                                dynamic_ncols=True,
                            ) as mega_pbar:
                                try:
                                    mega_summary = self._generate(
                                        prompt=mega_prompt,
                                        stream=True,
                                        token_pbar=mega_pbar,
                                    )
                                finally:
                                    self._gen_config.max_tokens = original_max_tokens
                            tokens_mega_summary = len(
                                self.model.tokenize(mega_summary.encode("utf-8"))
                            )
                            logger.warning(
                                f"Предыдущие саммари сжаты до: {tokens_mega_summary}. Экономия {((1 - (tokens_mega_summary / history_tokens)) * 100):.1f}%"
                            )

                            rolling_summary = [f"[СЖАТАЯ ИСТОРИЯ]: {mega_summary}"]
                            history = "\n".join(rolling_summary)

                        combined_context = f"Текущее саммари предыдущих чанков:\n{history}\n\nПоследний абзац предыдущей главы:\n{last_tail}"

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
                            response = self._generate(
                                prompt=self._build_prompt(
                                    chunk=chunk,
                                    cluster_topik=topik,
                                    previous_context=combined_context,
                                ),
                                stream=True,
                                token_pbar=chunk_token_pbar,
                            )
                            logger.debug(
                                f"Сгенерирован новый чанк конспекта, его длина: {
                                    len(self.model.tokenize(response.encode('utf-8')))
                                }."
                            )
                            # Сохраняем ответ
                            conspect[topik].append(response)

                            original_max_tokens = self._gen_config.max_tokens
                            self._gen_config.max_tokens = max_tokens_for_summary
                            summary_prompt = [
                                {
                                    "role": "system",
                                    "content": self.prompts[
                                        self._app_config.agent_name
                                    ]["summary_synthesizer"],
                                },
                                {
                                    "role": "user",
                                    "content": f"Выпиши САМОЕ ГЛАВНОЕ из этого текста:\n{response}",
                                },
                            ]
                            # УРОВЕНЬ 4: Генерация summary для следующего прохода агента
                            with tqdm(
                                total=max_tokens_for_summary,
                                desc="Генерация токенов для саммари",
                                unit="токен",
                                colour=ColoursForTqdm.fifth_level,
                                leave=False,
                                position=3,
                                file=sys.stdout,
                                dynamic_ncols=True,
                            ) as summary_token_pbar:
                                try:
                                    summary = self._generate(
                                        prompt=summary_prompt,
                                        stream=True,
                                        token_pbar=summary_token_pbar,
                                    )
                                finally:
                                    self._gen_config.max_tokens = original_max_tokens
                                logger.debug(
                                    f"Сгенерирован новый саммари конспекта, его длина: {
                                        len(
                                            self.model.tokenize(summary.encode('utf-8'))
                                        )
                                    }."
                                )

                        # Обновляем саммари
                        rolling_summary.append(f"[{topik}]: {summary}")
                        last_tail = " ".join(
                            response.split()[-(self._gen_config.max_tokens // 5) :]
                        )

                        # Обновляем бар чанков (Уровень 2)
                        chunk_pbar.update(1)

                # Обновляем бар глобальных кластеров (Уровень 1)
                pbar.update(1)

        logger.info("Генерация финального конспекта завершена.")

        out_filepath = self._safe_result_out_line(
            output_dict=conspect,
            stage="06_synthesizer/",
            file_name="conspect.json",
            session_dir=self.session_dir,
        )

        logger.success(f"Финальный конспект сохранен: {out_filepath}")
        return out_filepath
