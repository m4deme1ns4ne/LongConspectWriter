from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
from loguru import logger
import time
from tqdm import tqdm
import os
import sys
from os import PathLike
import torch
from src.core.utils import (
    TqdmTokenStreamer,
    TextsSplitter,
    log_retry_attempt,
    LoadPrompts,
    bad_words_id_generate,
)
from src.agents.base_agent import BaseLLMAgent
from tenacity import retry, stop_after_attempt, wait_fixed
from src.configs.ai_configs import AppLLMConfig, LLMGenConfig, LLMInitConfig
from dataclasses import asdict


class AgentPlanner(BaseLLMAgent):
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
            f"Инициализация агента Drafter (Модель: {self._init_config.pretrained_model_name_or_path}, Устройство: {self._init_config.device_map})"
        )

        logger.info(f"Загрузка {self._init_config.pretrained_model_name_or_path} в память...")
        self.model = AutoModelForCausalLM.from_pretrained(**asdict(self._init_config))
        logger.info(f"Модель {self._init_config.pretrained_model_name_or_path} загружена.")

        logger.info(f"Загрузка токенайзера для {self._init_config.pretrained_model_name_or_path} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._init_config.pretrained_model_name_or_path)
        logger.info(f"Токенайзер для {self._init_config.pretrained_model_name_or_path} загружен.")

        self.prompts = LoadPrompts.load_prompts(self._app_config.prompt_path)
        self.system_prompt, self.user_template = (
            self._load_prompts()
        )


    def _load_prompts(self) -> tuple:
        pass

    def _build_prompt(self) -> str:
        pass

    def _load_quant_config(self) -> object:
        pass

    def _generate(self) -> str:
        pass

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)