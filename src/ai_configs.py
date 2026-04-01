from dataclasses import dataclass
import torch
import gc
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from typing import Any
from transformers import AutoTokenizer
from loguru import logger


@dataclass
class AIModelConfig:
    """Родительский класс конфига для всех ИИ моделей"""

    pass


@dataclass
class LLMModelConfig(AIModelConfig):
    """Класс конфига для всех LLM моделей"""

    model: str
    torch_dtype: str | None = None
    device_map: str | None = None
    max_new_tokens: int = 500
    repetition_penalty: float = (1.15)
    temperature: float = (0.1)
    top_p: float = (0.9)
    do_sample: bool = True

    def __post_init__(self) -> None:
        if self.device_map is None:
            self.device_map = "auto" if torch.cuda.is_available() else "cpu"

        if self.torch_dtype is None:
            self.torch_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )


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


class VRamCleaner:
    @staticmethod
    def unload_model(model: Any) -> None:
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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
    @classmethod
    def get_vram_usage(cls):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            return f"{allocated:.0f} MB / {total:.0f} MB"
        return "CPU"
