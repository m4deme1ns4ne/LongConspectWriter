from dataclasses import dataclass
from os import PathLike
import torch
import yaml
from pathlib import Path


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
    torch_dtype: torch.dtype | None = None
    device_map: dict[str, int] | str | None = None
    agent_name: str | None = None
    prompt: Path | None = None

    def __post_init__(self) -> None:
        if self.device_map is None:
            self.device_map = {"": 0} if torch.cuda.is_available() else "cpu"

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


class LoadPrompts:
    @staticmethod
    def load_prompts(file_path: str | PathLike) -> dict[str, dict[str, str]]:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
