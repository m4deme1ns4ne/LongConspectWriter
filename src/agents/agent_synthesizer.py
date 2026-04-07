import os
from os import PathLike
from tqdm import tqdm
import time
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from transformers.generation.streamers import BaseStreamer
from src.agents.base_agent import BaseLLMAgent
from src.core.utils import (
    TqdmTokenStreamer,
    log_retry_attempt,
    LoadPrompts,
    bad_words_id_generate,
)
from tenacity import retry, stop_after_attempt, wait_fixed
from src.configs.ai_configs import AppLLMConfig, LLMGenConfig, LLMInitConfig
from dataclasses import asdict


class AgentSynthesizer(BaseLLMAgent):
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
            f"Инициализация агента Synthesizer (Модель: {self._init_config.pretrained_model_name_or_path}, Устройство: {self._init_config.device_map})"
        )

        logger.info(f"Загрузка {self._init_config.pretrained_model_name_or_path} в память...")
        self.model = AutoModelForCausalLM.from_pretrained(**asdict(self._init_config))
        logger.info(f"Модель {self._init_config.pretrained_model_name_or_path} загружена.")

        logger.info(f"Загрузка токенайзера для {self._init_config.pretrained_model_name_or_path} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._init_config.pretrained_model_name_or_path)
        logger.info(f"Токенайзер для {self._init_config.pretrained_model_name_or_path} загружен.")

        self.prompts = LoadPrompts.load_prompts(self._app_config.prompt_path)
        self.system_prompt, self.user_template = self._load_prompts()
        self.bad_words_ids = bad_words_id_generate(self.tokenizer)


    def _load_prompts(self) -> tuple[str]:
        system_prompt = self.prompts[self._app_config.agent_name]["system_prompt"]
        user_template = self.prompts[self._app_config.agent_name]["user_template"]
        return system_prompt, user_template

    def _build_prompt(self, draft_text: str) -> str:
        user_prompt = self.user_template.format(draft_text=draft_text)
        full_prompt = (
            f"{self.system_prompt}\n"
            f"{user_prompt}\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        return full_prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
    def _generate(self, prompt: str, streamer: BaseStreamer | None = None) -> str:
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        context_length = model_inputs.input_ids.shape[-1]
        logger.debug(f"Длина входного контекста: {context_length} токенов.")

        logger.info("Начало генерации финального конспекта...")
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                **asdict(self._gen_config),
                streamer=streamer,
                bad_words_ids=self.bad_words_ids,
            )[0]

        response = self.tokenizer.decode(
            output[context_length:], skip_special_tokens=True
        )

        return response

    def run(self, chunk_conspects_path: str | PathLike) -> str:
        with open(chunk_conspects_path, "r", encoding="utf-8") as file:
            draft_text = file.read()

        with tqdm(
            total=self._gen_config.max_new_tokens,
            desc="Генерация токенов",
            unit="токен",
            colour="blue",
            file=sys.stdout,
            dynamic_ncols=True,
        ) as token_pbar:
            streamer = TqdmTokenStreamer(token_pbar)
            final_text = self._generate(
                prompt=self._build_prompt(draft_text=draft_text),
                streamer=streamer,
            )

        logger.info("Генерация финального конспекта завершена.")

        timestamp = int(time.time())
        safe_model_name = self._init_config.pretrained_model_name_or_path.replace("/", "_")
        pure_draft_name = os.path.basename(chunk_conspects_path)
        out_filepath = os.path.join(
            self._app_config.output_dir,
            f"{safe_model_name}-{pure_draft_name}-{timestamp}.md",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "x", encoding="utf-8") as f:
            f.write(final_text)
        logger.success(f"Финальный конспект сохранен: {out_filepath}")
        return out_filepath
