from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm
from transformers.generation.streamers import BaseStreamer
from src.bad_words import BAD_WORDS
import yaml
from os import PathLike


class TextsSplitter:
    @staticmethod
    def split_text(text: str, model_name: str, chunk_size: int = 2000) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        )

        chunks = splitter.split_text(text)
        logger.info(
            f"Текст успешно разбит на {len(chunks)} фрагментов по {chunk_size} токенов каждый."
        )
        return chunks


class TqdmTokenStreamer(BaseStreamer):
    def __init__(self, pbar: tqdm) -> None:
        self.pbar = pbar
        self.is_first = True

    @staticmethod
    def _count_tokens(value: object) -> int:
        if hasattr(value, "numel"):
            return int(value.numel())

        if isinstance(value, (list, tuple)):
            count = 0
            for item in value:
                if isinstance(item, (list, tuple)):
                    count += len(item)
                else:
                    count += 1
            return count or 1

        return 1

    def put(self, value: object) -> None:
        if self.is_first:
            self.is_first = False
            return

        self.pbar.update(self._count_tokens(value))

    def end(self) -> None:
        if self.pbar:
            self.pbar.close()


def log_retry_attempt(retry_state):
    """Хук для tenacity, чтобы логировать неудачные попытки через loguru."""
    exception = retry_state.outcome.exception()
    logger.warning(
        f"Сбой выполнения. Попытка {retry_state.attempt_number}. "
        f"Ожидание {retry_state.next_action.sleep} сек. Ошибка: {exception}"
    )


def bad_words_id_generate(tokenizer) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    bad_words_ids: list[list[int]] = []

    for word in BAD_WORDS:
        # Токенизируем строку в список int без служебных токенов
        ids: list[int] = tokenizer.encode(word, add_special_tokens=False)

        # Пропускаем если токенизатор вернул пустой список
        if not ids:
            continue

        # Преобразуем в tuple для хранения в set (list не хешируемый)
        ids_tuple: tuple[int, ...] = tuple(ids)

        # Пропускаем дубликаты
        if ids_tuple in seen:
            continue

        seen.add(ids_tuple)
        bad_words_ids.append(ids)

    return bad_words_ids


class LoadPrompts:
    @staticmethod
    def load_prompts(file_path: str | PathLike) -> dict[str, dict[str, str]]:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
