import os
from src.ai_configs import (
    STTModelConfig,
    AIModelConfig,
    LLMModelConfig,
    LoadPrompts,
    TextsSplitter,
    VRamUsage,
)
from src.transcribing import FasterWhisper
from src.agent_drafter import AgentDrafter
from loguru import logger
from transformers import AutoTokenizer
import time
from tqdm import tqdm


def main():
    # stt_model_config = STTModelConfig(model_size="large-v3-turbo")
    # faster_whisper = FasterWhisper(stt_model_config)
    # faster_whisper.transcribing(
    #     audio_file_path=r"data\example-audio\Защита информации. Лекция 1..m4a"
    # )

    # 1. Загрузка текста транскрибации
    audio_file = (
        r"data\example-transcrib\large-v3-turbo-cuda-float16-Защита инф-1774820035.txt"
    )
    pure_audio_file_name = os.path.basename(audio_file)

    with open(audio_file, "r", encoding="utf-8") as file:
        transcrib = file.read()
        logger.info(f"Общая длина транскрибации: {len(transcrib)} символов")

    # 2. Нарезка на чанки
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    transcrib_chunks = TextsSplitter.split_text(text=transcrib, model_name=model_name)

    # 3. Подготовка промптов и инициализация
    prompts = LoadPrompts.load_prompts(r"src\prompts\drafter.yaml")
    system_prompt = prompts["drafter"]["system_prompt"]
    user_template = prompts["drafter"]["user_template"]

    logger.info(f"Старт инициализации. VRAM сейчас: {VRamUsage.get_vram_usage()}")
    agent_drafter_config = LLMModelConfig(model=model_name)
    agent_drafter = AgentDrafter(agent_drafter_config)
    logger.info(f"После инициализации модели. VRAM: {VRamUsage.get_vram_usage()}")

    final_drafts = []
    previous_summary = "Это начало лекции, предыдущего контекста нет."

    with tqdm(
        total=len(transcrib_chunks),
        unit="чанк",
        desc="Конспекты",
        colour="green",
    ) as pbar:
        for i, chunk in enumerate(transcrib_chunks):

            user_prompt = user_template.format(
                text=chunk, previous_summary=previous_summary
            )

            response = agent_drafter.generate(
                system_prompt=system_prompt, user_prompt=user_prompt
            )

            final_drafts.append(response)

            previous_summary = response

            pbar.update(1)

    monolith_draft = "\n\n---\n\n".join(final_drafts)

    safe_model_name = model_name.replace("/", "_")

    timestamp = int(time.time())
    out_filepath = os.path.join(
        "data",
        "example-conspect",
        f"{safe_model_name}-{pure_audio_file_name}-{timestamp}.txt",
    )

    with open(out_filepath, "w", encoding="utf-8") as f:
        f.write(monolith_draft)

    logger.success(f"✅ Финальный конспект сохранен по пути: {out_filepath}")


if __name__ == "__main__":
    main()
