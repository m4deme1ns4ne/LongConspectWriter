import os
from tqdm import tqdm
import time
from os import PathLike

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger
from src.ai_configs import (
    InitConfig,
    LoadPrompts,
    TqdmTokenStreamer,
    GenConfig,
    LLMAgent,
)
from dataclasses import asdict
from transformers.generation.streamers import BaseStreamer


class AgentSynthesizer(LLMAgent):
    def __init__(self, init_config: InitConfig, gen_config: GenConfig):
        self._init_config = init_config
        self._gen_config = gen_config
        logger.success(
            f"Инициализация агента Drafter (Synthesizer: {self._init_config.model}, Устройство: {self._init_config.device_map})"
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self._init_config.torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        logger.info(f"Загрузка {self._init_config.model} в память...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._init_config.model,
            quantization_config=quant_config,
            device_map=self._init_config.device_map,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        logger.success(f"Модель {self._init_config.model} загружена!")

        logger.info(f"Загрузка токинайзера для {self._init_config.model} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._init_config.model)
        logger.success(f"Загрузка токинайзера для {self._init_config.model} заверщена!")

        self.system_prompt, self.user_template = self._load_prompts()

    def _load_prompts(self):
        prompts = LoadPrompts.load_prompts(self._init_config.prompt)
        system_prompt = prompts[self._init_config.agent_name]["system_prompt"]
        user_template = prompts[self._init_config.agent_name]["user_template"]
        return system_prompt, user_template

    def _build_prompt(self, draft_text: str):
        user_promt = self.user_template.format(draft_text=draft_text)
        full_prompt = (
            f"{self.system_prompt}\n"
            f"{user_promt}\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        return full_prompt

    def _generate(self, prompt: str, streamer: BaseStreamer = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        context_length = inputs.input_ids.shape[-1]
        logger.info(f"Длина входного контекста: {context_length} токенов.")

        logger.info("Начало генерации финального конспекта...")
        output = self.model.generate(
            **inputs, **asdict(self._gen_config), streamer=streamer
        )[0]

        response = self.tokenizer.decode(
            output[context_length:], skip_special_tokens=True
        )

        return response

    @staticmethod
    def _normalize_draft_path(
        chunk_conspects_path: str | PathLike | tuple[list[str], str | PathLike],
    ) -> str:
        if isinstance(chunk_conspects_path, tuple):
            if len(chunk_conspects_path) != 2:
                raise TypeError(
                    "chunk_conspects_path tuple must contain "
                    "(draft_chunks, draft_file_path)."
                )
            _, chunk_conspects_path = chunk_conspects_path

        if isinstance(chunk_conspects_path, PathLike):
            return os.fspath(chunk_conspects_path)

        if isinstance(chunk_conspects_path, str):
            return chunk_conspects_path

        raise TypeError(
            "chunk_conspects_path must be a path string or "
            "(draft_chunks, draft_file_path) tuple."
        )

    def run(
        self, chunk_conspects_path: str | PathLike | tuple[list[str], str | PathLike]
    ):
        chunk_conspects_path = self._normalize_draft_path(chunk_conspects_path)

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

                # Передаем весь монолит разом
                final_text = self._generate(
                    prompt=self._build_prompt(draft_text=draft_text),
                    streamer=streamer,
                )

        logger.success("Генерация всех частей завершена!")

        timestamp = int(time.time())
        safe_model_name = self._init_config.model.replace("/", "-")
        pure_draft_name = os.path.basename(chunk_conspects_path)
        out_filepath = os.path.join(
            "data",
            "example-conspect",
            f"{safe_model_name}-{pure_draft_name}-{timestamp}.md",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "x", encoding="utf-8") as f:
            f.write(final_text)

        logger.success(f"Идеальный конспект сохранен в {out_filepath}!")

        return out_filepath
