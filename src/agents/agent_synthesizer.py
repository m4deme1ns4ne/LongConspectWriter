import os
from tqdm import tqdm
import time
import sys
from pathlib import Path
import json

from loguru import logger
from src.core.base import BaseTransformersAgent
from src.core.utils import TqdmTokenStreamer, TextsSplitter, bad_words_id_generate
from src.configs.bad_words import BAD_WORDS_SYNTHESIZER


class AgentSynthesizer(BaseTransformersAgent):
    def __init__(self, init_config, gen_config, app_config):
        super().__init__(init_config, gen_config, app_config)

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            global_clusters = json.load(file)

        conspect = {topik: [] for topik, _ in global_clusters.items()}

        previous_context = "Это начало лекции, предыдущего контекста нет."

        bad_words_ids = bad_words_id_generate(
            tokenizer=self.tokenizer, bad_words=BAD_WORDS_SYNTHESIZER
        )

        # УРОВЕНЬ 1: Цикл по глобальным темам (Кластерам)
        with tqdm(
            total=len(global_clusters),
            unit="кластер",
            desc="Темы",
            colour="green",
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for topik, clusters in global_clusters.items():
                full_text = " ".join(clusters)
                split_clusters: list[str] = TextsSplitter.split_text_to_tokens(
                    text=full_text,
                    model_name=self._init_config.pretrained_model_name_or_path,
                    chunk_size=int(self._gen_config.max_new_tokens * 0.5),
                    chunk_overlap=0,
                )

                # УРОВЕНЬ 2: Цикл по кускам текста внутри темы
                with tqdm(
                    total=len(split_clusters),
                    desc="Чанки",
                    unit="чанк",
                    colour="red",
                    leave=False,
                    position=1,
                    file=sys.stdout,
                    dynamic_ncols=True,
                ) as chunk_pbar:
                    for chunk in split_clusters:
                        # УРОВЕНЬ 3: Цикл генерации токенов для одного куска
                        with tqdm(
                            total=self._gen_config.max_new_tokens,
                            desc="Генерация токенов",
                            unit="токен",
                            colour="blue",
                            leave=False,
                            position=2,
                            file=sys.stdout,
                            dynamic_ncols=True,
                        ) as token_pbar:
                            streamer = TqdmTokenStreamer(token_pbar)
                            response = self._generate(
                                prompt=self._build_prompt(
                                    chunk=chunk,
                                    cluster_topik=topik,
                                    previous_context=previous_context,
                                ),
                                streamer=streamer,
                                bad_words_ids=bad_words_ids,
                            )

                        # Обновляем контекст и сохраняем ответ
                        conspect[topik].append(response)
                        previous_context = " ".join(
                            response.split()[-(self._gen_config.max_new_tokens // 3) :]
                        )

                        # Обновляем бар чанков (Уровень 2)
                        chunk_pbar.update(1)

                # Обновляем бар глобальных кластеров (Уровень 1)
                pbar.update(1)

        logger.info("Генерация финального конспекта завершена.")

        timestamp = int(time.time())
        safe_model_name = self._init_config.pretrained_model_name_or_path.replace(
            "/", "_"
        )
        pure_draft_name = os.path.basename(path)
        out_filepath = os.path.join(
            self._app_config.output_dir,
            f"{safe_model_name}-{pure_draft_name}-{timestamp}.json",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as file:
            json.dump(conspect, file, ensure_ascii=False, indent=4)
        logger.success(f"Финальный конспект сохранен: {out_filepath}")
        return out_filepath
