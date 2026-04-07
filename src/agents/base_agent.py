from abc import ABC, abstractmethod
from loguru import logger
from src.core.vram_manager import VRamCleaner


class BaseAgent(ABC):
    @abstractmethod
    def run(self) -> object:
        pass

    def __enter__(self) -> "BaseAgent":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        logger.debug(f"Очистка памяти от агента {self.__class__.__name__}...")
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        VRamCleaner.empty_vram(caller_name=self.__class__.__name__)


class BaseLLMAgent(BaseAgent):
    @abstractmethod
    def _generate(self) -> str:
        pass

    @abstractmethod
    def _load_prompts(self) -> tuple:
        pass

    @abstractmethod
    def _build_prompt(self) -> str:
        pass


class BaseSTTAgent(BaseAgent):
    pass
