"""Общие вспомогательные утилиты для этапов пайплайна LongConspectWriter.

Модуль содержит разбиение текста, загрузку промптов, настройку ретраев,
логирование времени выполнения, проверку выходных путей и загрузку пакетов
конфигураций, используемых core-этапами и агентами.
"""

from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from os import PathLike
import os
from contextlib import contextmanager, redirect_stderr
import functools
import time
from razdel import sentenize
import sys
import ctypes
import io
import warnings
import logging
from llama_cpp import llama_log_callback, llama_log_set
from dataclasses import dataclass
from src.configs.configs import AgentConfigBundle
from tenacity import stop_after_attempt, wait_fixed, retry
from tenacity import RetryCallState
from typing import Callable, Any, TypeVar, ParamSpec
from collections.abc import Generator
import tqdm

P = ParamSpec("P")
R = TypeVar("R")


class TextsSplitter:
    """Разбивает текст лекции на предложения и токен-размерные фрагменты."""

    @staticmethod
    def split_text_to_sentences(text: str) -> list[str]:
        """Разбивает транскрипт или тело кластера на строки предложений.

        Args:
            text (str): Текст транскрипта или кластера из этапов кластеризации
                LongConspectWriter.

        Returns:
            list[str]: Упорядоченные тексты предложений, извлеченные из входа.
        """
        sentences = list(sentenize(text))
        return [sentence.text for sentence in sentences]

    @staticmethod
    def split_text_to_tokens(
        text: str,
        tokenizer: Callable[[bytes], Any],
        chunk_size: int = 2000,
        chunk_overlap: int = 0,
    ) -> list[str]:
        """Разбивает текст на чанки, измеряемые переданным токенизатором модели.

        Args:
            text (str): Текст лекции, который должен уложиться в контекст LLM.
            tokenizer (Callable[[bytes], Any]): Callable токенизатора активной модели
                для оценки длины закодированного чанка.
            chunk_size (int): Максимальная токен-длина каждого чанка.
            chunk_overlap (int): Перекрытие между соседними чанками.

        Returns:
            list[str]: Упорядоченные текстовые чанки, готовые для prompt агента.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(tokenizer(x.encode("utf-8"))),
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        )

        chunks = splitter.split_text(text)
        logger.debug(
            f"Текст успешно разбит на {len(chunks)} фрагментов по {chunk_size} токенов каждый."
        )
        return chunks


def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Хук для tenacity, чтобы логировать неудачные попытки через loguru.

    Args:
        retry_state (RetryCallState): Состояние упавшего вызова агента или утилиты,
            который ретраится внутри пайплайна.

    Returns:
        None: Функция только логирует retry-диагностику.
    """
    exception = retry_state.outcome.exception()
    logger.warning(
        f"Сбой выполнения. Попытка {retry_state.attempt_number}. "
        f"Ожидание {retry_state.next_action.sleep} сек. Ошибка: {exception}"
    )


class LoadPrompts:
    """Загружает YAML-словари промптов для агентов."""

    @staticmethod
    def load_prompts(file_path: str | PathLike) -> dict[str, dict[str, str]]:
        """Читает YAML-файл промптов агента.

        Args:
            file_path (str | PathLike): Путь к конфигурации prompt, используемой
                агентом LongConspectWriter.

        Returns:
            dict[str, dict[str, str]]: Распарсенное дерево prompt с ключами по именам агентов
            и секциям prompt.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)


modify_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=log_retry_attempt,
    reraise=True,
)


def log_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    """Декорирует метод этапа пайплайна логированием времени выполнения.

    Args:
        func (Callable[P, R]): Метод этапа, обычно ``run``, длительность которого
            должна быть видна в логах.

    Returns:
        Callable[P, R]: Обернутый callable с неизмененным возвращаемым значением.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Запускает обернутый callable пайплайна и логирует прошедшее время.

        Args:
            *args (P.args): Позиционные аргументы для обернутого метода этапа.
            **kwargs (P.kwargs): Именованные аргументы для обернутого метода этапа.

        Returns:
            R: Неизмененный результат обернутого callable.
        """
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


def check_path_is(func: Callable[P, R]) -> Callable[P, R]:
    """Проверяет, что этап пайплайна вернул непустой path-like результат.

    Args:
        func (Callable[P, R]): Метод перехода пайплайна, который должен вернуть
            путь к артефакту для следующего этапа LongConspectWriter.

    Returns:
        Callable[P, R]: Обернутая функция, которая завершает процесс при пустом пути.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Запускает обернутый этап и проверяет, что он создал path-like значение.

        Args:
            *args (P.args): Позиционные аргументы для обернутого метода пайплайна.
            **kwargs (P.kwargs): Именованные аргументы для обернутого метода пайплайна.

        Returns:
            R: Непустой результат обернутого callable.
        """
        out_path = func(*args, **kwargs)
        if out_path is None:
            logger.critical(f"{func.__qualname__} сгенерировал пустой путь.")
            sys.exit(1)
        return out_path

    return wrapper


@dataclass
class ColoursForTqdm:
    """Цветовые константы для вложенных progress bar LongConspectWriter."""

    first_level: str = "#002b00"
    second_level: str = "#005f00"
    third_level: str = "#00af00"
    fourth_level: str = "#00ff00"
    fifth_level: str = "#afff00"


def load_agent_bundle(
    yaml_path: str | PathLike,
    cls_init_config: Callable[..., Any] | None = None,
    cls_gen_config: Callable[..., Any] | None = None,
    cls_app_config: Callable[..., Any] | None = None,
) -> AgentConfigBundle:
    """Загружает пакет конфигурации агента из YAML в указанные типы dataclass.

    Args:
        yaml_path (str | PathLike): Путь к YAML-конфигу агента.
        cls_init_config (Callable[..., Any] | None): Dataclass или фабрика для
            секции инициализации модели.
        cls_gen_config (Callable[..., Any] | None): Dataclass или фабрика для
            секции генерации.
        cls_app_config (Callable[..., Any] | None): Dataclass или фабрика для
            секции применения в пайплайне.

    Returns:
        AgentConfigBundle: Пакет конфигурации, который используется конструктором соответствующего агента.
    """
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    init_cfg = (
        cls_init_config(**config.get("init_config", {})) if cls_init_config else None
    )
    gen_cfg = cls_gen_config(**config.get("gen_config", {})) if cls_gen_config else None
    app_cfg = cls_app_config(**config.get("app_config", {})) if cls_app_config else None

    return AgentConfigBundle(
        init_config=init_cfg, gen_config=gen_cfg, app_config=app_cfg
    )


_llama_noop_log_cb: Any = None


def configure_logger() -> None:
    """Настраивает loguru, фильтры Python-предупреждений и sys.unraisablehook.

    Вызывать один раз при старте пайплайна. Подавляет шум от huggingface_hub,
    sentence_transformers, transformers и llama_cpp ctypes-коллбэков.
    """
    logger.remove()
    logger.add(
        lambda msg: tqdm.tqdm.write(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        catch=False
    )

    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)

    global _llama_noop_log_cb
    _llama_noop_log_cb = llama_log_callback(lambda level, msg, ud: None)
    llama_log_set(_llama_noop_log_cb, ctypes.c_void_p(0))

    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    _prev_unraisablehook = sys.unraisablehook

    def _filtered_unraisablehook(unraisable: Any) -> None:
        if "llama_log_callback" in repr(getattr(unraisable, "object", None) or ""):
            return
        _prev_unraisablehook(unraisable)

    sys.unraisablehook = _filtered_unraisablehook


@contextmanager
def suppress_c_stderr() -> Generator[None, None, None]:
    """Подавляет C-уровневый и Python-уровневый вывод в stderr.

    Перенаправляет файловый дескриптор 2 (stderr) в os.devnull для C-кода
    и временно подменяет sys.stderr для Python-уровневого вывода (например,
    llama_context предупреждения от llama_cpp через ctypes callback).
    Дескриптор гарантированно восстанавливается даже при исключении.

    Yields:
        None: Контекстный менеджер не возвращает значения.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        with redirect_stderr(io.StringIO()):
            yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        if _llama_noop_log_cb is not None:
            llama_log_set(_llama_noop_log_cb, ctypes.c_void_p(0))
