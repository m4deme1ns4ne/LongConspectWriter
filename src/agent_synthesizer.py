import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger

def synthesize_final_conspectus():
    # 1. Читаем наш сырой черновик от Драфтера
    draft_file_path = r"data\example-conspect\Qwen_Qwen2.5-7B-Instruct-large-v3-turbo-cuda-float16-Защита инф-1774820035-1111.txt"
    
    logger.info(f"Чтение файла черновиков: {draft_file_path}")
    with open(draft_file_path, "r", encoding="utf-8") as f:
        draft_text = f.read()

    # 2. Инициализация LongWriter
    model_id = "THUDM/LongWriter-llama3.1-8b"
    logger.info("Загрузка токенайзера LongWriter...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    logger.info("Загрузка модели LongWriter (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model.eval()

    # 3. ФОРМИРОВАНИЕ ПРОМПТА ДЛЯ СИНТЕЗАТОРА
    # Даем жесткую инструкцию на русском языке
    system_prompt = (
        "Ты — профессиональный академический редактор и составитель учебников. "
        "Тебе переданы черновые заметки с одной университетской лекции по IT и криптографии. "
        "В этих заметках много дубликатов, так как они создавались кусками.\n"
        "ТВОЯ ЗАДАЧА:\n"
        "1. Объединить все заметки в один цельный, логичный и красивый конспект (лонгрид) в формате Markdown.\n"
        "2. БЕЗЖАЛОСТНО удалить все смысловые повторы (если про RSA сказано три раза — оставь одно самое полное описание).\n"
        "3. Сгруппировать текст по смысловым блокам (например: 'Организационные вопросы', 'Работа с файловыми системами', 'Введение в криптографию', 'Математический аппарат').\n"
        "4. Строго соблюдать академический стиль. Исключи слова 'Лектор', 'учитель', 'автор лекции'. Пиши безлично: 'Рассматривается...', 'Алгоритм Евклида заключается в...'.\n"
        "5. Сохранить абсолютно все математические термины, программы (FAR, Maple) и факты.\n"
        "Не пиши вводных фраз вроде 'Вот ваш конспект'. Выводи сразу готовый Markdown-текст."
    )

    user_prompt = f"ЧЕРНОВЫЕ ЗАМЕТКИ ДЛЯ ОБРАБОТКИ:\n\n{draft_text}"

    # Собираем промпт в формате Llama-3
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    logger.info("Токенизация промпта (этот кусок текста займет много токенов)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    context_length = inputs.input_ids.shape[-1]
    logger.info(f"Длина входного контекста: {context_length} токенов.")

    logger.info("Начало генерации финального конспекта. Это может занять 5-10 минут...")
    
    # 4. Генерация
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4096, # Даем модели пространство для создания большого лонгрида
            do_sample=True,
            temperature=0.2,     # Низкая температура для строгой логики
            top_p=0.9,
            repetition_penalty=1.1, # Легкая защита от зацикливаний
        )[0]

    logger.success("Генерация завершена!")
    
    # 5. Декодирование и сохранение
    response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    
    final_output_path = r"data\example-conspect\FINAL_MONOLITH_CONSPECTUS.md"
    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write(response)

    logger.success(f"Идеальный конспект сохранен в {final_output_path}!")

synthesize_final_conspectus()
