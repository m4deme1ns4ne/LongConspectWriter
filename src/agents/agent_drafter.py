from loguru import logger
import time
from tqdm import tqdm
import os
import sys
from src.core.utils import (
    TqdmTokenStreamer,
    TextsSplitter,
)
from src.core.base import BaseTransformersAgent
from pathlib import Path


class AgentDrafter(BaseTransformersAgent):
    def __init__(self, init_config, gen_config, app_config):
        super().__init__(init_config, gen_config, app_config)

    def run(self, path_transcrib: Path) -> str:
        with open(path_transcrib, "r", encoding="utf-8") as file:
            transcrib: str = file.read()

        token_count = len(self.tokenizer.encode(transcrib, add_special_tokens=False))

        len_original_sentences = len(TextsSplitter.split_text_to_sentences(transcrib))

        logger.info(f"Кол-во предложений до чистки: {len_original_sentences}.")
        logger.info(f"Кол-во токенов до чистки: {token_count}.")

        transcrib_chunks = TextsSplitter.split_text_to_tokens(
            text=transcrib,
            model_name=self._init_config.pretrained_model_name_or_path,
            chunk_size=int(self._gen_config.max_new_tokens * 0.85),
            chunk_overlap=0,
        )

        final_drafts = []
        previous_context = "Это начало лекции, предыдущего контекста нет."
        ignored_chunks = 0

        compression_ratio = []

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
                if "[NO_CONTEXT]" in response:
                    ignored_chunks += 1
                    continue

                orig_tokens = len(
                    self.tokenizer.encode(chunk, add_special_tokens=False)
                )
                clean_tokens = len(
                    self.tokenizer.encode(response, add_special_tokens=False)
                )

                compression_ratio.append(
                    {"orig_tokens": orig_tokens, "clean_tokens": clean_tokens}
                )

                final_drafts.append(response)
                previous_context = " ".join(response.split()[-40:])

        if not final_drafts:
            logger.warning("Drafter не нашел ни одного содержательного чанка.")
            return ""
        
        final_drafts = " ".join(final_drafts)

        for i, chunk in enumerate(compression_ratio, start=1):
            orig_tokens = chunk["orig_tokens"]
            clean_tokens = chunk["clean_tokens"]
            if orig_tokens > 0:
                compression = (1 - (clean_tokens / orig_tokens)) * 100
                logger.debug(
                    f"Чанк {i}: {orig_tokens} -> {clean_tokens} токенов | Сжатие: {compression:.1f}%"
                )

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
            f.write(final_drafts)

        logger.success(f"Очищенная транскрибация сохранена: {out_filepath}")
        if ignored_chunks:
            logger.debug(
                f"Пропущено пустых чанков: {ignored_chunks}, потенциальная экономия: "
                f"{ignored_chunks * self._gen_config.max_new_tokens} токенов."
            )
        else:
            logger.debug(f"Пропущеных чанков не обнаружено.")

        len_final_drafts = len(TextsSplitter.split_text_to_sentences(final_drafts))
        current_token_count = len(
            self.tokenizer.encode(final_drafts, add_special_tokens=False)
        )
        defference_tokens = (
            token_count - current_token_count
            if token_count > current_token_count
            else 0
        )
        defference_sentences = (
            len_original_sentences - len_final_drafts
            if len_original_sentences > len_final_drafts
            else 0
        )

        logger.info(f"Кол-во предложений после чистки: {len_final_drafts}.")
        logger.info(f"Кол-во токенов после чистки: {current_token_count}.")
        logger.info(f"Экономия в предложениях: {defference_sentences}.")
        logger.info(f"Экономия в токенах: {defference_tokens}.")

        return out_filepath
