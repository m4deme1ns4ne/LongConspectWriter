import os
from tqdm import tqdm
import time

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
            bnb_4bit_compute_dtype=torch.bfloat16,
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

    def run(self, chunk_conspects_path):

        logger.info(f"Чтение файла черновиков: {chunk_conspects_path}")
        with open(chunk_conspects_path, "r", encoding="utf-8") as f:
            draft_text = f.read()

        prompts = LoadPrompts.load_prompts(r"src\prompts\drafter.yaml")

        system_prompt = prompts["synthesizer"]["system_prompt"]
        user_prompt = prompts["synthesizer"]["user_template"].format(
            draft_text=draft_text
        )

        full_prompt = (
            f"{system_prompt}\n"
            f"{user_prompt}\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        self.model.eval()

        with tqdm(
            total=self._gen_config.max_new_tokens,
            desc="Генерация токенов",
            unit="токен",
            colour="green",
            position=0,
        ) as token_pbar:
            with torch.no_grad():
                streamer = TqdmTokenStreamer(token_pbar)
                response = self._generate(
                    prompt=full_prompt,
                    streamer=streamer,
                )

        logger.success("Генерация завершена!")

        timestamp = int(time.time())
        safe_model_name = self._init_config.model.replace("/", "-")
        pure_draft_name = os.path.basename(chunk_conspects_path)
        out_filepath = os.path.join(
            "data",
            "example-conspect",
            f"{safe_model_name}-{pure_draft_name}-{timestamp}",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "x", encoding="utf-8") as f:
            f.write(response)

        logger.success(f"Идеальный конспект сохранен в {out_filepath}!")

        return out_filepath
