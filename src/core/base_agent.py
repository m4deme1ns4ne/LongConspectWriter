from abc import ABC, abstractmethod
from loguru import logger


class BaseAgent(ABC):
    @abstractmethod
    def run(self) -> object:
        pass


class BaseLLMAgent(BaseAgent):
    @abstractmethod
    def _generate(self) -> str:
        pass

    @abstractmethod
    def _load_prompts(self) -> tuple[str, str]:
        pass

    @abstractmethod
    def _build_prompt(self) -> str:
        pass

    def __enter__(self) -> "BaseLLMAgent":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        logger.info(f"Очистка памяти от агента {self.__class__.__name__}...")
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer


class BaseSTTAgent(BaseAgent):
    pass
