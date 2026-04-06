from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers.generation.streamers import BaseStreamer
from loguru import logger
import time
from tqdm import tqdm
import os
import sys
from os import PathLike
import torch
from src.utils import TqdmTokenStreamer, TextsSplitter, log_retry_attempt, LoadPrompts, bad_words_id_generate
from src.core.base_agent import BaseLLMAgent
from tenacity import retry, stop_after_attempt, wait_fixed
from src.ai_configs import LLMInitConfig, LLMGenConfig


class AgentDrafter(BaseLLMAgent):
    def __init__(self, init_config: LLMInitConfig, gen_config: LLMGenConfig) -> None:
        self._init_config = init_config
        self._gen_config = gen_config
        logger.info(
            f"Инициализация агента Drafter (Модель: {self._init_config.model}, Устройство: {self._init_config.device_map})"
        )

        quant_config = self._load_quant_config()

        logger.info(f"Загрузка {self._init_config.model} в память...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._init_config.model,
            **self._init_config.model_kwargs(),
            quantization_config=quant_config,
        )
        logger.info(f"Модель {self._init_config.model} загружена.")

        logger.info(f"Загрузка токенайзера для {self._init_config.model} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._init_config.model)
        logger.info(f"Токенайзер для {self._init_config.model} загружен.")

        self.prompts = LoadPrompts.load_prompts(self._init_config.prompt)
        self.system_prompt, self.user_template, self.negative_prompt = (
            self._load_prompts()
        )

        self.bad_words_ids = bad_words_id_generate(self.tokenizer)
        self.negative_inputs = self.tokenizer(
            [self.negative_prompt], return_tensors="pt"
        ).to(self.model.device)

    def _load_quant_config(self) -> BitsAndBytesConfig:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self._init_config.torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return quant_config

    def _load_prompts(self) -> tuple[str, str]:
        system_prompt = self.prompts[self._init_config.agent_name]["system_prompt"]
        user_template = self.prompts[self._init_config.agent_name]["user_template"]
        negative_prompt = self.prompts[self._init_config.agent_name]["negative_prompt"]
        return system_prompt, user_template, negative_prompt

    def _build_prompt(self, chunk: str, previous_summary: str) -> str:
        user_prompt = self.user_template.format(
            draft_text=chunk, previous_summary=previous_summary
        )
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return full_prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        before_sleep=log_retry_attempt,
        reraise=True,
    )
    def _generate(self, prompt: str, streamer: BaseStreamer | None = None) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(
            self.model.device
        )
        generation_kwargs = self._gen_config.generation_kwargs(
            exclude={
                "negative_prompt_ids",
                "negative_prompt_attention_mask",
                "bad_words_ids",
            }
        )

        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                **generation_kwargs,
                streamer=streamer,
                negative_prompt_ids=self.negative_inputs.input_ids,
                negative_prompt_attention_mask=self.negative_inputs.attention_mask,
                bad_words_ids=self.bad_words_ids,
            )
        output = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, output)
        ]

        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return response

    def run(self, path_transcrib: str | PathLike) -> str:
        pure_transcrib_file_name = os.path.basename(path_transcrib)

        with open(path_transcrib, "r", encoding="utf-8") as file:
            transcrib = file.read()
        token_count = len(self.tokenizer.encode(transcrib, add_special_tokens=False))
        logger.info(
            f"Общая длина транскрибации: {len(transcrib)} символов или {token_count} токенов."
        )

        # 2. Нарезка на чанки
        transcrib_chunks = TextsSplitter.split_text(
            text=transcrib, model_name=self._init_config.model
        )

        final_drafts: list[str] = []
        previous_summary = "Это начало лекции, предыдущего контекста нет."
        ignored_chunks = 0

        with tqdm(
            total=len(transcrib_chunks),
            unit="чанк",
            desc="Конспекты",
            colour="green",
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in transcrib_chunks:
                with tqdm(
                    total=self._gen_config.max_new_tokens,
                    desc="Генерация токенов",
                    unit="токен",
                    colour="blue",
                    leave=False,
                    position=1,
                    file=sys.stdout,
                    dynamic_ncols=True,
                ) as token_pbar:
                    streamer = TqdmTokenStreamer(token_pbar)
                    response = self._generate(
                        prompt=self._build_prompt(
                            chunk=chunk, previous_summary=previous_summary
                        ),
                        streamer=streamer,
                    )
                pbar.update(1)
                if "[NO CONTENT FOUND]" in response:
                    ignored_chunks += 1
                    continue

                final_drafts.append(response)
                previous_summary = " ".join(response.split()[-40:])

        if not final_drafts:
            logger.warning("Drafter не нашел ни одного содержательного чанка.")
            return ""

        len_final_drafts = len(final_drafts)

        monolith_draft = "\n\n---\n\n".join(final_drafts)

        safe_model_name = self._init_config.model.replace("/", "_")

        timestamp = int(time.time())
        out_filepath = os.path.join(
            "data",
            "example-mini-conspect",
            f"{safe_model_name}-{pure_transcrib_file_name}-{timestamp}.txt",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            f.write(monolith_draft)
        logger.success(
            f"Финальные мини-конспекты сохранены (валидных чанков: {len_final_drafts}). "
        )
        if ignored_chunks:
            logger.info(
                f"Пропущено пустых чанков: {ignored_chunks}, потенциальная экономия: "
                f"{ignored_chunks * self._gen_config.max_new_tokens} токенов."
            )

        return out_filepath
