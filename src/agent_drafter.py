from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.ai_configs import LLMModelConfig
from loguru import logger
import torch


class AgentDrafter:
    def __init__(self, config: LLMModelConfig):
        """
        Сейчас ваш класс AgentDrafter загружает модель и
        токенайзер в память каждый раз при вызове метода generate.
        Это занимает время и память. В идеале их стоит загружать один
        раз в __init__, чтобы генерировать ответы мгновенно при последующих вызовах.
        """

        self._config = config
        logger.success(
            f"Инициализация агента Drafter (Модель: {self._config.model}, Устройство: {self._config.device_map})"
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self._config.torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        logger.info(f"Загрузка {self._config.model} в память...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._config.model,
            torch_dtype=self._config.torch_dtype,
            device_map=self._config.device_map,
            quantization_config=quant_config,
            attn_implementation="sdpa",
        )
        logger.success(f"Модель {self._config.model} загружена!")

        logger.info(f"Загрузка токинайзера для {self._config.model} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._config.model)
        logger.success(f"Загрузка токинайзера для {self._config.model} заверщена!")

    def generate(self, user_prompt: str, system_prompt: str = None) -> str:
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

        generated_ids = self.model.generate(
            max_new_tokens=self._config.max_new_tokens, **model_inputs
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response
