from loguru import logger
import time
from tqdm import tqdm
import os
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
from src.core.utils import ColoursForTqdm
import json
import ast


class AgentLocalPlanner(BaseLlamaCppAgent):
    def __init__(self, init_config, gen_config, app_config, session_dir: Path):
        self.session_dir = session_dir
        super().__init__(init_config, gen_config, app_config)

    def _chunking(self, text: list, max_tokens: int):

        final_batches = []
        current_batch = []
        current_tokens = 0

        for chunk in text:
            tokenized_chunk = len(self.model.tokenize(chunk.encode("utf-8")))
            len_chunk_tokens = tokenized_chunk

            if len_chunk_tokens + current_tokens > max_tokens:
                current_batch = "\n\n------------------------\n\n".join(current_batch)
                final_batches.append(current_batch)

                current_batch = [chunk]
                current_tokens = len_chunk_tokens

            else:
                current_batch.append(chunk)
                current_tokens += len_chunk_tokens

        if current_batch:
            current_batch = "\n\n------------------------\n\n".join(current_batch)
            final_batches.append(current_batch)

        return final_batches

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            local_clusters = json.load(file)

        local_clusters = list(local_clusters.values())

        max_tokens = int(self._gen_config.max_tokens * 2)
        chunking_local_clusters = self._chunking(local_clusters, max_tokens)
        logger.info(
            f"Локальных кластеров разбитых на {max_tokens} токенов получилось: {len(chunking_local_clusters)}"
        )

        final_drafts = []
        ignored_chunks = 0

        with tqdm(
            total=len(chunking_local_clusters),
            unit="чанк",
            desc="Локальные кластеры",
            colour=ColoursForTqdm.first_level,
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in chunking_local_clusters:
                with tqdm(
                    total=self._gen_config.max_tokens,
                    desc="Генерация токенов",
                    unit="токен",
                    colour=ColoursForTqdm.second_level,
                    leave=False,
                    position=1,
                    file=sys.stdout,
                    dynamic_ncols=True,
                ) as token_pbar:
                    response = self._generate(
                        prompt=self._build_prompt(text=chunk),
                        stream=True,
                        token_pbar=token_pbar,
                    )
                pbar.update(1)
                if "[NO_TOPICS]" in response:
                    ignored_chunks += 1
                    continue

                final_drafts.append(response)

        logger.info("Генерация локальных заголовков завершена.")

        full_local_plan = " ".join(final_drafts)
        response_dict = {"answer_agent": full_local_plan}
        out_filepath = self._safe_result_out_line(
            output_dict=response_dict,
            stage="03_local_planners/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )
        return out_filepath


class AgentGlobalPlanner(BaseLlamaCppAgent):
    def __init__(self, init_config, gen_config, app_config, session_dir: Path):
        self.session_dir = session_dir
        super().__init__(init_config, gen_config, app_config)

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            local_plan = json.load(file)

        local_plan = local_plan["answer_agent"]
        with tqdm(
            total=self._gen_config.max_tokens,
            desc="Генерация токенов",
            unit="токен",
            colour="blue",
            leave=False,
            position=1,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as token_pbar:
            response = self._generate(
                prompt=self._build_prompt(text=local_plan),
                stream=True,
                token_pbar=token_pbar,
            )
        logger.info("Генерация глобальных заголовков завершена.")

        response_dict = ast.literal_eval(response)

        out_filepath = self._safe_result_out_line(
            output_dict=response_dict,
            stage="04_global_planners/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.success(f"Глобальные заголовки сохранены: {out_filepath}")
        return out_filepath
