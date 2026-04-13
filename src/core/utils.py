from loguru import logger
from tqdm import tqdm
from transformers.generation.streamers import BaseStreamer
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from os import PathLike
import functools
import time
from razdel import sentenize


class TextsSplitter:
    @staticmethod
    def split_text_to_sentences(text: str) -> list[str]:
        sentences = list(sentenize(text))
        return [sentence.text for sentence in sentences]

    @staticmethod
    def split_text_to_tokens(
        text: str, model_name: str, chunk_size: int = 2000, chunk_overlap: int = 0
    ) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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


def bad_words_id_generate(tokenizer, bad_words: list[str]) -> list[list[int]]:
    seen = set()
    bad_words_ids = []
    for word in bad_words:
        # Токенизируем строку в список int без служебных токенов
        ids = tokenizer.encode(word, add_special_tokens=False)

        # Пропускаем если токенизатор вернул пустой список
        if not ids:
            continue

        # Преобразуем в tuple для хранения в set (list не хешируемый)
        ids_tuple = tuple(ids)

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


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.success(
            f"[{func.__qualname__}] завершён за "
            f"{int(hours)}ч {int(minutes)}м {seconds:.1f}с"
        )
        return result

    return wrapper


SEPARATOR = "\n\n------------------------\n\n"
