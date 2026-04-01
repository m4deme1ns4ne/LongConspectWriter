import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger
from src.ai_configs import LLMModelConfig, LoadPrompts


class AgentSynthesizer:
    def __init__(self, config: LLMModelConfig):
        self._config = config
        logger.success(
            f"Инициализация агента Drafter (Модель: {self._config.model}, Устройство: {self._config.device_map})"
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        logger.info(f"Загрузка {self._config.model} в память...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._config.model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        logger.success(f"Модель {self._config.model} загружена!")

        logger.info(f"Загрузка токинайзера для {self._config.model} в память...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._config.model)
        logger.success(f"Загрузка токинайзера для {self._config.model} заверщена!")

    def synthesize(self, chunk_conspects_path):

        logger.info(f"Чтение файла черновиков: {chunk_conspects_path}")
        with open(chunk_conspects_path, "r", encoding="utf-8") as f:
            draft_text = f.read()

        self.model.eval()

        # 3. ФОРМИРОВАНИЕ ПРОМПТА ДЛЯ СИНТЕЗАТОРА
        # Даем жесткую инструкцию на русском языке
        prompts = LoadPrompts.load_prompts(r"src\prompts\drafter.yaml")
        system_prompt = prompts["synthesizer"]["system_prompt"]

        user_prompt = f"ЧЕРНОВЫЕ ЗАМЕТКИ ДЛЯ ОБРАБОТКИ:\n\n{draft_text}"

        # Собираем промпт в формате Llama-3
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        logger.info("Токенизация промпта (этот кусок текста займет много токенов)...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        context_length = inputs.input_ids.shape[-1]
        logger.info(f"Длина входного контекста: {context_length} токенов.")

        logger.info(
            "Начало генерации финального конспекта. Это может занять 5-10 минут..."
        )

        # 4. Генерация
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=4096,  # Даем модели пространство для создания большого лонгрида
                do_sample=True,
                temperature=0.2,  # Низкая температура для строгой логики
                top_p=0.9,
                repetition_penalty=1.1,  # Легкая защита от зацикливаний
            )[0]

        logger.success("Генерация завершена!")

        # 5. Декодирование и сохранение
        response = self.tokenizer.decode(
            output[context_length:], skip_special_tokens=True
        )

        final_output_path = r"data\example-conspect\FINAL_MONOLITH_CONSPECTUS.md"

        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        with open(final_output_path, "x", encoding="utf-8") as f:
            f.write(response)

        logger.success(f"Идеальный конспект сохранен в {final_output_path}!")

        return final_output_path
