from abc import ABC, abstractmethod
from loguru import logger
from src.core.vram_manager import VRamCleaner
from src.core.utils import log_execution_time, LoadPrompts, log_retry_attempt
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.configs.ai_configs import AppLLMConfig, LLMGenConfig, LLMInitConfig
from dataclasses import asdict
import torch
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers.generation.streamers import BaseStreamer


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
    Базовый класс. Классы которые его наследуют обязуются иметь метод run
    """

    @abstractmethod
    def run(self) -> object:
        pass


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
    def _generate(self) -> str:
        pass

    @abstractmethod
    def _build_prompt(self) -> str:
        pass


class BaseTransformersAgent(BaseLLMAgent):
    def __init__(
        self,
        init_config: LLMInitConfig,
        gen_config: LLMGenConfig,
        app_config: AppLLMConfig,
    ) -> None:
        self._init_config = init_config
        self._gen_config = gen_config
        self._app_config = app_config
        logger.info(
            f"Инициализация {self.__class__.__name__} (Модель: {self._init_config.pretrained_model_name_or_path}, Устройство: {self._init_config.device_map})"
        )

        logger.info(
            f"Загрузка {self._init_config.pretrained_model_name_or_path} в память..."
        )
        self.model = AutoModelForCausalLM.from_pretrained(**asdict(self._init_config))
        logger.info(
            f"Модель {self._init_config.pretrained_model_name_or_path} загружена."
        )

        logger.info(
            f"Загрузка токенайзера для {self._init_config.pretrained_model_name_or_path} в память..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._init_config.pretrained_model_name_or_path
        )
        logger.info(
            f"Токенайзер для {self._init_config.pretrained_model_name_or_path} загружен."
        )

        self.prompts = LoadPrompts.load_prompts(self._app_config.prompt_path)
        self.system_prompt = self.prompts[self._app_config.agent_name]["system_prompt"]
        self.user_template = self.prompts[self._app_config.agent_name]["user_template"]
        self.negative_prompt = self.prompts[self._app_config.agent_name][
            "negative_prompt"
        ]
        if self.negative_prompt and self._gen_config.guidance_scale > 1.0:
            logger.info(
                f"Негативный промпт загружен успешно для агента {self.__class__.__name__}"
            )
        elif self.negative_prompt and self._gen_config.guidance_scale <= 1.0:
            logger.warning(
                f"Негативный промпт обнаружен но не загружен для агента: {self.__class__.__name__}, так как guidance_scale == {self._gen_config.guidance_scale}"
            )
        else:
            logger.warning(
                f"Негативный промпт не обнаружен для агента: {self.__class__.__name__}"
            )

        logger.debug(
            f"Параметры запуска агента {self.__class__.__name__}: {self._init_config}"
        )
        logger.debug(
            f"Параметры генерации агента {self.__class__.__name__}: {self._gen_config}"
        )
        logger.debug(
            f"Параметры использования агента {self.__class__.__name__}: {self._app_config}"
        )

    def _build_prompt(self, **kwargs) -> str:
        user_prompt = self.user_template.format(**kwargs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
    def _generate(
        self,
        prompt: str,
        streamer: BaseStreamer | None = None,
        bad_words_ids: list[list[int]] | None = None,
    ) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(
            self.model.device
        )
        if self.negative_prompt:
            negative_prompt_ids = (
                self.tokenizer([self.negative_prompt], return_tensors="pt")
                .to(self.model.device)
                .input_ids
            )
        else:
            negative_prompt_ids = None
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                **asdict(self._gen_config),
                streamer=streamer,
                bad_words_ids=bad_words_ids,
                negative_prompt_ids=negative_prompt_ids,
            )
        output = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, output)
        ]

        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return response


class BaseSTTAgent(BaseAgent):
    pass


class Clustering(Trackable, Base):
    pass
