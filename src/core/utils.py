from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from os import PathLike
import functools
import time
from razdel import sentenize
import sys
from dataclasses import dataclass
from src.configs.configs import AgentConfigBundle


class TextsSplitter:
    @staticmethod
    def split_text_to_sentences(text: str) -> list[str]:
        # cleaned_text = TextsSplitter._sanitize_drafter_text(text)
        sentences = list(sentenize(text))
        return [sentence.text for sentence in sentences]

    # @staticmethod
    # def _sanitize_drafter_text(text: str) -> str:
    #     """
    #     Нормализатор "грязного" текста от LLM.
    #     Восстанавливает пунктуацию для корректной работы razdel.
    #     """
    #     # 1. Защита от "потерянных" точек перед переносом строки.
    #     # Если перед \n нет знака препинания (., !, ?, :, ;), ставим точку принудительно.
    #     text = re.sub(r"(?<![\.\!\?\:\;])(\s*\n)", r".\1", text)

    #     # 2. Гарантия точки после семантических тегов.
    #     text = re.sub(r"(\[[А-ЯЁA-Z]+\])(?!\s*[\.\!\?\,\:])", r"\1.", text)

    #     # 3. Базовая очистка мусорных пробелов, чтобы не плодить пустые токены
    #     text = re.sub(r"[ \t]{2,}", " ", text)

    #     return text

    @staticmethod
    def split_text_to_tokens(
        text: str, tokenizer: str, chunk_size: int = 2000, chunk_overlap: int = 0
    ) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(tokenizer(x.encode("utf-8"))),
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        )

        chunks = splitter.split_text(text)
        logger.info(
            f"Текст успешно разбит на {len(chunks)} фрагментов по {chunk_size} токенов каждый."
        )
        return chunks


def log_retry_attempt(retry_state):
    """Хук для tenacity, чтобы логировать неудачные попытки через loguru."""
    exception = retry_state.outcome.exception()
    logger.warning(
        f"Сбой выполнения. Попытка {retry_state.attempt_number}. "
        f"Ожидание {retry_state.next_action.sleep} сек. Ошибка: {exception}"
    )


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
        logger.debug(
            f"[{func.__qualname__}] завершён за "
            f"{int(hours)}ч {int(minutes)}м {seconds:.1f}с"
        )
        return result

    return wrapper


def check_path_is(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        out_path = func(*args, **kwargs)
        if out_path is None:
            logger.critical(f"{func.__qualname__} сгенерировал пустой путь.")
            sys.exit(1)
        return out_path

    return wrapper


@dataclass
class ColoursForTqdm:
    first_level: str = "#002b00"
    second_level: str = "#005f00"
    third_level: str = "#00af00"
    fourth_level: str = "#00ff00"
    fifth_level: str = "#afff00"


def load_agent_bundle(
    yaml_path,
    cls_init_config,
    cls_gen_config,
    cls_app_config,
) -> AgentConfigBundle:
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    init_config = cls_init_config(**config.get("init_config", {}))
    gen_config = cls_gen_config(**config.get("gen_config", {}))
    app_config = cls_app_config(**config.get("app_config", {}))

    return AgentConfigBundle(
        init_config=init_config, gen_config=gen_config, app_config=app_config
    )
