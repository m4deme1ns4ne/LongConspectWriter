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


class Trackable(ABC):
    @abstractmethod
    def run(self) -> object:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            cls.run = log_execution_time(cls.run)


class BaseAgent(Trackable):
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
        self.system_prompt, self.user_template = self._load_prompts()

        logger.debug(
            f"Параметры запуска агента {self.__class__.__name__}: {self._init_config}"
        )
        logger.debug(
            f"Параметры генерации агента {self.__class__.__name__}: {self._gen_config}"
        )

    def _load_prompts(self) -> tuple:
        system_prompt = self.prompts[self._app_config.agent_name]["system_prompt"]
        user_template = self.prompts[self._app_config.agent_name]["user_template"]
        return system_prompt, user_template

    def _build_prompt(self, **kwargs) -> str:
        self.user_prompt = self.user_template.format(**kwargs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                **asdict(self._gen_config),
                streamer=streamer,
                bad_words_ids=bad_words_ids,
            )
        output = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, output)
        ]

        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return response


class BaseSTTAgent(BaseAgent):
    pass


class Clustering(Trackable):
    pass


class Compession(Trackable):
    pass
