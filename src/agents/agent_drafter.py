from loguru import logger
import time
from tqdm import tqdm
import os
import sys
from src.core.utils import (
    TqdmTokenStreamer,
    TextsSplitter,
)
from src.agents.base_agent import BaseTransformersAgent
from pathlib import Path
from src.core.utils import SEPARATOR


class AgentDrafter(BaseTransformersAgent):
    def __init__(self):
        super().__init__()

    def run(self, path_transcrib: Path) -> str:
        with open(path_transcrib, "r", encoding="utf-8") as file:
            transcrib = file.read()
        token_count = len(self.tokenizer.encode(transcrib, add_special_tokens=False))
        logger.info(
            f"Общая длина транскрибации: {len(transcrib)} символов или {token_count} токенов."
        )

        transcrib_chunks = TextsSplitter.split_text(
            text=transcrib, model_name=self._init_config.pretrained_model_name_or_path
        )

        final_drafts = []
        previous_context = "Это начало лекции, предыдущего контекста нет."
        ignored_chunks = 0

        with tqdm(
            total=len(transcrib_chunks),
            unit="чанк",
            desc="Конспекты",
            colour="green",
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in transcrib_chunks:
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
                        prompt=self._build_prompt(
                            chunk=chunk, previous_context=previous_context
                        ),
                        streamer=streamer,
                    )
                pbar.update(1)
                if "[NO_TOPICS]" in response:
                    ignored_chunks += 1
                    continue

                final_drafts.append(response)
                previous_context = " ".join(response.split()[-40:])

        if not final_drafts:
            logger.warning("Drafter не нашел ни одного содержательного чанка.")
            return ""

        len_final_drafts = len(final_drafts)

        monolith_draft = f"\n\n{SEPARATOR}\n\n".join(final_drafts)

        safe_model_name = self._init_config.pretrained_model_name_or_path.replace(
            "/", "_"
        )

        pure_transcrib_file_name = os.path.basename(path_transcrib)

        timestamp = int(time.time())
        out_filepath = os.path.join(
            self._app_config.output_dir,
            f"{safe_model_name}-{pure_transcrib_file_name}-{timestamp}.txt",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            f.write(monolith_draft)
        logger.success(
            f"Финальные мини-конспекты сохранены (валидных чанков: {len_final_drafts}). "
        )
        if ignored_chunks:
            logger.info(
                f"Пропущено пустых чанков: {ignored_chunks}, потенциальная экономия: "
                f"{ignored_chunks * self._gen_config.max_new_tokens} токенов."
            )

        return out_filepath
