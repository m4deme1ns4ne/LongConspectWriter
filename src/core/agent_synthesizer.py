import os
from os import PathLike
from tqdm import tqdm
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger
from src.ai_configs import InitConfig, LoadPrompts, GenConfig
from dataclasses import asdict
from transformers.generation.streamers import BaseStreamer
from src.core.base_agent import BaseLLMAgent
from src.utils import TqdmTokenStreamer, log_retry_attempt
from tenacity import retry, stop_after_attempt, wait_fixed
import src.utils as utils


class AgentSynthesizer(BaseLLMAgent):
    def __init__(self, init_config: InitConfig, gen_config: GenConfig) -> None:
        self._init_config = init_config
        self._gen_config = gen_config
        logger.success(
            f"Инициализация агента Synthesizer (Модель: {self._init_config.model}, Устройство: {self._init_config.device_map})"
        )

        quant_config = self._load_quant_config()

        logger.info(f"Загрузка {self._init_config.model} в память...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._init_config.model,
            quantization_config=quant_config,
            device_map=self._init_config.device_map,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        logger.success(f"Модель {self._init_config.model} загружена!")

        logger.info(f"Загрузка токенайзера для {self._init_config.model} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._init_config.model)
        logger.success(f"Загрузка токенайзера для {self._init_config.model} завершена!")

        self.system_prompt, self.user_template = self._load_prompts()
        self.bad_words_ids = self._tokenize_bad_words_ids()

    def _load_quant_config(self) -> BitsAndBytesConfig:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self._init_config.torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return quant_config

    def _load_prompts(self) -> tuple[str, str, str]:
        prompts = LoadPrompts.load_prompts(self._init_config.prompt)
        system_prompt = prompts[self._init_config.agent_name]["system_prompt"]
        user_template = prompts[self._init_config.agent_name]["user_template"]
        return system_prompt, user_template

    def _build_prompt(self, draft_text: str) -> str:
        user_prompt = self.user_template.format(draft_text=draft_text)
        full_prompt = (
            f"{self.system_prompt}\n"
            f"{user_prompt}\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        return full_prompt
    
    def _tokenize_bad_words_ids(self) -> list:
        bad_words_ids = utils.bad_words_id_generate(self.tokenizer)
        return bad_words_ids

    @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(5),
            before_sleep=log_retry_attempt,
            reraise=True
        )
    def _generate(self, prompt: str, streamer: BaseStreamer | None = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        context_length = inputs.input_ids.shape[-1]
        logger.info(f"Длина входного контекста: {context_length} токенов.")

        logger.info("Начало генерации финального конспекта...")
        output = self.model.generate(
            **inputs,
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
        ) as token_pbar:
            with torch.no_grad():
                streamer = TqdmTokenStreamer(token_pbar)
                final_text = self._generate(
                    prompt=self._build_prompt(draft_text=draft_text),
                    streamer=streamer,
                )

        logger.success("Генерация всех частей завершена!")

        timestamp = int(time.time())
        safe_model_name = self._init_config.model.replace("/", "_")
        pure_draft_name = os.path.basename(chunk_conspects_path)
        out_filepath = os.path.join(
            "data",
            "example-conspect",
            f"{safe_model_name}-{pure_draft_name}-{timestamp}.md",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "x", encoding="utf-8") as f:
            f.write(final_text)
        return out_filepath
