from dataclasses import dataclass
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from transformers import AutoTokenizer
from loguru import logger
import functools
import time
import gc
from transformers.generation.streamers import BaseStreamer
from abc import ABC, abstractmethod


@dataclass
class AIModelConfig:
    """Родительский класс конфига для всех ИИ моделей"""

    pass


@dataclass
class LLMModelConfig(AIModelConfig):
    """Конфиг для всех LLM моделей"""

    pass


@dataclass
class InitConfig(LLMModelConfig):
    """Конфиг только для загрузки весов в память"""

    model: str
    torch_dtype: str | None = None
    device_map: str | None = None

    def __post_init__(self) -> None:
        if self.device_map is None:
            self.device_map = "auto" if torch.cuda.is_available() else "cpu"

        if self.torch_dtype is None:
            self.torch_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )


@dataclass
class GenConfig(LLMModelConfig):
    """Конфиг только для параметров генерации (kwargs)"""

    max_new_tokens: int = 500
    repetition_penalty: float = 1.15
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class STTModelConfig(AIModelConfig):
    """Класс конфига для всех STT моделей"""

    model_size: str
    device: str | None = None
    compute_type: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"


class Agent(ABC):
    @abstractmethod
    def run(self):
        pass


class LLMAgent(Agent):
    @abstractmethod
    def _generate(self):
        pass

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"Очистка памяти агента {self.__class__.__name__}...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

class STTAgent(Agent):
    pass


class TextsSplitter:
    @staticmethod
    def split_text(text: str, model_name: str, chunk_size: int = 2000) -> list:
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


class LoadPrompts:
    @staticmethod
    def load_prompts(file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)


class VRamUsage:
    @staticmethod
    def get_vram_usage() -> str:
        if not torch.cuda.is_available():
            return "CPU"

        try:
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            allocated = torch.cuda.memory_allocated(device_index) / (1024**2)
            reserved = torch.cuda.memory_reserved(device_index) / (1024**2)
            total = props.total_memory / (1024**2)
            return (
                f"allocated={allocated:.0f} MB, "
                f"reserved={reserved:.0f} MB / {total:.0f} MB"
            )
        except Exception as exc:
            return f"GPU (usage unavailable: {exc})"


class _VRamCleaner:
    @staticmethod
    def empty_vram(caller_name: str | None = None) -> None:
        owner = caller_name or "unknown"
        gc.collect()

        if not torch.cuda.is_available():
            logger.info(f"[{owner}] GPU не использовался.")
            return

        try:
            before = VRamUsage.get_vram_usage()

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()

            torch.cuda.empty_cache()

            after = VRamUsage.get_vram_usage()
            logger.success(f"[{owner}] VRAM успешно очищена: {before} -> {after}")
        except Exception:
            logger.exception(f"[{owner}] Не удалось корректно очистить VRAM.")


def decorator_v_ram_cleaner(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__qualname__
        start = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(f"[{name}] Упал через {elapsed:.1f} сек.")
            raise
        else:
            elapsed = time.perf_counter() - start
            logger.success(f"[{name}] Завершен за {elapsed:.1f} сек.")
            return result
        finally:
            _VRamCleaner.empty_vram(caller_name=name)

    return wrapper


class TqdmTokenStreamer(BaseStreamer):
    def __init__(self, pbar):
        self.pbar = pbar
        self.is_first = True

    def put(self, value):
        if self.is_first:
            self.is_first = False
            return

        self.pbar.update(1)

    def end(self):
        if self.pbar:
            self.pbar.close()
