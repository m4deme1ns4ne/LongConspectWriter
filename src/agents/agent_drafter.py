from loguru import logger
import time
from tqdm import tqdm
import os
import sys
from src.core.utils import TextsSplitter, ColoursForTqdm
from src.core.base import BaseLlamaCppAgent
from pathlib import Path


class AgentDrafterLlama(BaseLlamaCppAgent):
    def __init__(self, init_config, gen_config, app_config):
        super().__init__(init_config, gen_config, app_config)

    def run(self, path_transcrib: Path) -> str:
        with open(path_transcrib, "r", encoding="utf-8") as file:
            transcrib: str = file.read()

        token_count = len(self.model.tokenize(transcrib.encode("utf-8")))

        len_original_sentences = len(TextsSplitter.split_text_to_sentences(transcrib))

        logger.info(f"Кол-во предложений до чистки: {len_original_sentences}.")
        logger.info(f"Кол-во токенов до чистки: {token_count}.")

        transcrib_chunks = TextsSplitter.split_text_to_tokens(
            text=transcrib,
            tokenizer=self.model.tokenize,
            chunk_size=int(self._gen_config.max_tokens * 0.85),
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
            colour=ColoursForTqdm.first_level,
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in transcrib_chunks:
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
                        prompt=self._build_prompt(
                            chunk=chunk, previous_context=previous_context
                        ),
                        stream=True,
                        token_pbar=token_pbar,
                    )
                pbar.update(1)
                response = response.strip()
                if "[NO_CONTEXT]" == response:
                    ignored_chunks += 1
                    continue
                response = response.replace("[NO_CONTEXT]", "").strip()

                orig_tokens = len(self.model.tokenize(chunk.encode("utf-8")))
                clean_tokens = len(self.model.tokenize(response.encode("utf-8")))

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

        self._safe_result_out_line(output_text=...)

        if ignored_chunks:
            logger.debug(
                f"Пропущено пустых чанков: {ignored_chunks}, потенциальная экономия: "
                f"{ignored_chunks * self._gen_config.max_tokens} токенов."
            )
        else:
            logger.debug("Пропущеных чанков не обнаружено.")

        len_final_drafts = len(TextsSplitter.split_text_to_sentences(final_drafts))
        current_token_count = len(self.model.tokenize(final_drafts.encode("utf-8")))
        difference_tokens = (
            token_count - current_token_count
            if token_count > current_token_count
            else 0
        )
        difference_sentences = (
            len_original_sentences - len_final_drafts
            if len_original_sentences > len_final_drafts
            else 0
        )

        logger.info(f"Кол-во предложений после чистки: {len_final_drafts}.")
        logger.info(f"Кол-во токенов после чистки: {current_token_count}.")
        logger.info(f"Экономия в предложениях: {difference_sentences}.")
        logger.info(f"Экономия в токенах: {difference_tokens}.")

        return out_filepath
