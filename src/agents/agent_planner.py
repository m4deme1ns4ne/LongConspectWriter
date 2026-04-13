from loguru import logger
import time
from tqdm import tqdm
import os
import sys
from src.core.utils import TqdmTokenStreamer
from src.core.base import BaseTransformersAgent
from pathlib import Path
from src.core.utils import SEPARATOR


class AgentLocalPlanner(BaseTransformersAgent):
    def __init__(self, init_config, gen_config, app_config):
        super().__init__(init_config, gen_config, app_config)

    def _chunking(self, text: str, max_tokens: int):
        local_clusters = text.split(f"\n\n{SEPARATOR}\n\n")

        final_batches = []
        current_batch = []
        current_tokens = 0

        for chunk in local_clusters:
            tokenized_chunk = self.tokenizer(chunk, add_special_tokens=False)[
                "input_ids"
            ]
            len_chunk_tokens = len(tokenized_chunk)

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
            local_clusters = file.read()

        max_tokens = 2000
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
            colour="green",
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in chunking_local_clusters:
                with tqdm(
                    total=self._gen_config.max_new_tokens,
                    desc="Генерация токенов",
                    unit="токен",
                    colour="blue",
                    leave=False,
                    position=1,
                    file=sys.stdout,
                    dynamic_ncols=True,
                ) as token_pbar:
                    streamer = TqdmTokenStreamer(token_pbar)
                    response = self._generate(
                        prompt=self._build_prompt(text=chunk),
                        streamer=streamer,
                    )
                pbar.update(1)
                if "[NO_TOPICS]" in response:
                    ignored_chunks += 1
                    continue

                final_drafts.append(response)

        logger.info("Генерация локальных заголовков завершена.")

        final_drafts = f"\n\n{SEPARATOR}\n\n".join(final_drafts)

        timestamp = int(time.time())
        safe_model_name = self._init_config.pretrained_model_name_or_path.replace(
            "/", "_"
        )
        pure_draft_name = os.path.basename(path)
        out_path = os.path.join(
            self._app_config.output_dir,
            f"{safe_model_name}-{pure_draft_name}-{timestamp}.md",
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "x", encoding="utf-8") as f:
            f.write(final_drafts)
        logger.success(f"Локальные заголовки сохранены: {out_path}")
        return out_path


class AgentGlobalPlanner(BaseTransformersAgent):
    def __init__(self, init_config, gen_config, app_config):
        super().__init__(init_config, gen_config, app_config)

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            local_plan = file.read()

        local_plan = local_plan.replace(f"\n\n{SEPARATOR}\n\n", "\n")

        with tqdm(
            total=self._gen_config.max_new_tokens,
            desc="Генерация токенов",
            unit="токен",
            colour="blue",
            leave=False,
            position=1,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as token_pbar:
            streamer = TqdmTokenStreamer(token_pbar)
            response = self._generate(
                prompt=self._build_prompt(text=local_plan),
                streamer=streamer,
            )
        logger.info("Генерация глобальных заголовков завершена.")

        timestamp = int(time.time())
        safe_model_name = self._init_config.pretrained_model_name_or_path.replace(
            "/", "_"
        )
        pure_draft_name = os.path.basename(path)
        out_path = os.path.join(
            self._app_config.output_dir,
            f"{safe_model_name}-{pure_draft_name}-{timestamp}.json",
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "x", encoding="utf-8") as f:
            f.write(response)
        logger.success(f"Глобальные заголовки сохранены: {out_path}")
        return out_path
