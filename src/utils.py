from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm
from transformers.generation.streamers import BaseStreamer
from src.bad_words import BAD_WORDS


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
        logger.success(
            f"Текст успешно разбит на {len(chunks)} фрагментов по {chunk_size} токенов каждый."
        )
        return chunks


class TqdmTokenStreamer(BaseStreamer):
    def __init__(self, pbar: tqdm) -> None:
        self.pbar = pbar
        self.is_first = True

    def put(self, value: object) -> None:
        if self.is_first:
            self.is_first = False
            return

        self.pbar.update(1)

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
