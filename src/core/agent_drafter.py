from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
from loguru import logger
import time
from tqdm import tqdm
import os
from src.ai_configs import (
    LoadPrompts,
    TextsSplitter,
    TqdmTokenStreamer,
    InitConfig,
    GenConfig,
    LLMAgent,
)
import torch
from dataclasses import asdict


class AgentDrafter(LLMAgent):
    def __init__(self, init_config: InitConfig, gen_config: GenConfig):
        """
        Сейчас ваш класс AgentDrafter загружает модель и
        токенайзер в память каждый раз при вызове метода generate.
        Это занимает время и память. В идеале их стоит загружать один
        раз в __init__, чтобы генерировать ответы мгновенно при последующих вызовах.
        """
        self._init_config = init_config
        self._gen_config = gen_config
        logger.success(
            f"Инициализация агента Drafter (Модель: {self._init_config.model}, Устройство: {self._init_config.device_map})"
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
            torch_dtype=self._init_config.torch_dtype,
            device_map=self._init_config.device_map,
            quantization_config=quant_config,
            attn_implementation="sdpa",
        )
        logger.success(f"Модель {self._init_config.model} загружена!")

        logger.info(f"Загрузка токинайзера для {self._init_config.model} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._init_config.model)
        logger.success(f"Загрузка токинайзера для {self._init_config.model} заверщена!")

    def _generate(
        self, user_prompt: str, system_prompt: str = None, streamer: BaseStreamer = None
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **model_inputs, **asdict(self._gen_config), streamer=streamer
            )
        output = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, output)
        ]

        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return response

    def run(self, path_transcrib: str):
        pure_transcrib_file_name = os.path.basename(path_transcrib)

        with open(path_transcrib, "r", encoding="utf-8") as file:
            transcrib = file.read()
            logger.info(f"Общая длина транскрибации: {len(transcrib)} символов")

        # 2. Нарезка на чанки
        transcrib_chunks = TextsSplitter.split_text(
            text=transcrib, model_name=self._init_config.model
        )

        # 3. Подготовка промптов и инициализация
        prompts = LoadPrompts.load_prompts(r"src\prompts\drafter.yaml")
        system_prompt = prompts["drafter"]["system_prompt"]
        user_template = prompts["drafter"]["user_template"]

        final_drafts = []
        previous_summary = "Это начало лекции, предыдущего контекста нет."

        with tqdm(
            total=len(transcrib_chunks),
            unit="чанк",
            desc="Конспекты",
            colour="green",
            position=0,
        ) as pbar:
            for chunk in transcrib_chunks:

                user_prompt = user_template.format(
                    text=chunk, previous_summary=previous_summary
                )

                with tqdm(
                    total=self._gen_config.max_new_tokens,
                    desc="Генерация токенов",
                    unit="токен",
                    colour="blue",
                    leave=False,  # Убираем бар после завершения чанка
                    position=1,  # Рисуем под основным баром
                ) as token_pbar:

                    streamer = TqdmTokenStreamer(token_pbar)
                    response = self._generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        streamer=streamer,
                    )

                final_drafts.append(response)
                previous_summary = response
                pbar.update(1)

        monolith_draft = "\n\n---\n\n".join(final_drafts)

        safe_model_name = self._init_config.model.replace("/", "-")

        timestamp = int(time.time())
        out_filepath = os.path.join(
            "data",
            "example-mini-conspect",
            f"{safe_model_name}-{pure_transcrib_file_name}-{timestamp}.txt",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            f.write(monolith_draft)

        logger.success(f"✅ Финальные мини-конспекты сохранены по пути: {out_filepath}")

        return out_filepath
