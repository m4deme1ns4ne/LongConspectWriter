from abc import ABC, abstractmethod
from loguru import logger
from src.core.vram_manager import VRamCleaner
from src.core.utils import log_execution_time, LoadPrompts, log_retry_attempt
from src.configs.configs import (
    LLMInitConfig,
    LLMGenConfig,
    LLMAppConfig,
)
from dataclasses import asdict
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Any
from llama_cpp import Llama
from src.configs.configs import AppSTTConfig, STTGenConfig, STTInitConfig
from faster_whisper import WhisperModel
import json
from pathlib import Path
from datetime import datetime


class Trackable:
    """
    Класс для логирования времени выполениния метода run
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            cls.run = log_execution_time(cls.run)


class Base(ABC):
    """
    Базовый класс всех частей пайплайна. Классы которые его наследуют обязуются иметь метод run
    """

    @abstractmethod
    def run(self) -> str:
        pass

    def _safe_result_out_line(
        self,
        output_dict: dict[str, Any] | str,
        stage: str,
        file_name: str,
        session_dir: Path,
        extension: str = "json",
    ) -> Path:

        actual_stage_path = session_dir / stage

        actual_stage_path.mkdir(parents=True, exist_ok=True)

        output_file_path = actual_stage_path / file_name

        if extension == "json":
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(output_dict, file, ensure_ascii=False, indent=4)
        elif extension == "md":
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(str(output_dict))
        logger.success(
            f"Работа агента {self.__class__.__name__} сохранена в файл по пути: {output_file_path}"
        )
        return output_file_path


class BaseAgent(Trackable, Base):
    """
    Базовый класс. Наследует Trackable и Base. Классы которые его наследуют обязывают иметь метод run
    и при выходе из контексткого менеджера очищают ГПУ.
    """

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
    """
    Базовый класс. Наследуется от BaseAgent. Классы которые его наследуют обязывают иметь методы
    _generate, _load_prompts и _build_prompt.
    """

    @abstractmethod
    def _generate(self, prompt: Any, **kwargs) -> str:
        pass

    @abstractmethod
    def _build_prompt(self) -> Any:
        pass


class BaseLlamaCppAgent(BaseLLMAgent):
    def __init__(
        self,
        init_config: LLMInitConfig,
        gen_config: LLMGenConfig,
        app_config: LLMAppConfig,
    ) -> None:
        self._init_config = init_config
        self._gen_config = gen_config
        self._app_config = app_config
        logger.info(
            f"Инициализация {self.__class__.__name__} (Модель: {self._init_config.model_path})"
        )

        logger.info(f"Загрузка {self._init_config.model_path} в память...")
        self.model = Llama(**asdict(self._init_config))
        logger.info(f"Модель {self._init_config.model_path} загружена.")

        self.prompts = LoadPrompts.load_prompts(self._app_config.prompt_path)
        self.system_prompt = self.prompts[self._app_config.agent_name]["system_prompt"]
        self.user_template = self.prompts[self._app_config.agent_name]["user_template"]

        logger.debug(
            f"Параметры запуска агента {self.__class__.__name__}: {self._init_config}"
        )
        logger.debug(
            f"Параметры генерации агента {self.__class__.__name__}: {self._gen_config}"
        )
        logger.debug(
            f"Параметры использования агента {self.__class__.__name__}: {self._app_config}"
        )

    def _build_prompt(self, **kwargs) -> list[dict[str]]:
        user_prompt = self.user_template.format(**kwargs)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
    def _generate(
        self, prompt: list[dict[str, str]], stream: bool = False, token_pbar: Any = None
    ) -> str:
        if stream:
            response_text = ""
            generator = self.model.create_chat_completion(
                messages=prompt, stream=True, **asdict(self._gen_config)
            )
            for chunk in generator:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    text_chunk = delta["content"]
                    response_text += text_chunk
                    if token_pbar:
                        token_pbar.update(1)
            return response_text
        else:
            response = self.model.create_chat_completion(
                messages=prompt, stream=False, **asdict(self._gen_config)
            )
            return response["choices"][0]["message"]["content"]


class BaseSTTAgent(BaseAgent):
    def __init__(
        self,
        init_config: STTInitConfig,
        gen_config: STTGenConfig,
        app_config: AppSTTConfig,
    ) -> None:
        self._init_config = init_config
        self._gen_config = gen_config
        self._app_config = app_config
        logger.info(
            f"Инициализация агента STT (Модель: {self._init_config.model_size_or_path}, Устройство: {self._init_config.device})"
        )
        logger.info(f"Загрузка {self._init_config.model_size_or_path} в память...")
        self.model = WhisperModel(
            **asdict(self._init_config),
        )
        logger.info(f"Модель {self._init_config.model_size_or_path} загружена.")

        logger.debug(
            f"Параметры запуска агента {self.__class__.__name__}: {self._init_config}"
        )
        logger.debug(
            f"Параметры генерации агента {self.__class__.__name__}: {self._gen_config}"
        )
        logger.debug(
            f"Параметры использования агента {self.__class__.__name__}: {self._app_config}"
        )
        self.prompt = LoadPrompts.load_prompts(self._app_config.prompt_path)

        if self._app_config.the_subject_lecture is None:
            self.initial_prompt = self.prompt[self._app_config.agent_name]["universal"]
        else:
            self.initial_prompt = self.prompt[self._app_config.agent_name][
                self._app_config.the_subject_lecture
            ]


class BaseClusterizer(Trackable, Base):
    pass


class BasePipeline(Trackable, Base):
    def __post_init__(self) -> None:

        output_dir = Path(self.pipeline_config.output_dir)

        self.actual_session_dir = (
            output_dir / "runs" / datetime.now().strftime("%H.%M.%S___%Y.%m.%d")
        )
        self.actual_session_dir.mkdir(exist_ok=False, parents=True)

        logger.success(
            f"Папка актуальной сессии создана по пути: {self.actual_session_dir}"
        )
