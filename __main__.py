from src.pipeline import ConspectiusPipeline
from loguru import logger
from dotenv import load_dotenv
import os
from src.ai_configs import STTInitConfig, STTGenConfig, LLMInitConfig, LLMGenConfig, VadParametersConfig
from pathlib import Path
import yaml


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_pipeline_configs(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    vad_parameters_config = VadParametersConfig(**config["vad_parameters_config"])

    stt_init_config = STTInitConfig(**config["stt_init_config"])
    stt_gen_config = STTGenConfig(vad_parameters_config=vad_parameters_config ,**config["stt_gen_config"])
    drafter_init = LLMInitConfig(**config["drafter_init_config"])
    drafter_gen = LLMGenConfig(**config["drafter_gen_config"])
    synth_init = LLMInitConfig(**config["synthesizer_init_config"])
    synth_gen = LLMGenConfig(**config["synthesizer_gen_config"])

    return stt_init_config, stt_gen_config, drafter_init, drafter_gen, synth_init, synth_gen


def main() -> None:
    load_dotenv()

    # временное решение до тестов
    full_audio_file_path = (
        Path("data") / "example-audio" / "Лекция 1. Вещественные числа. Часть 1.mp3"
    )

    # временное решение до тестов
    litle_audio_file_path = (
        Path("data") / "example-audio" / "Лекция 1 (mp3cut.net).mp3"
    )

    configs = load_pipeline_configs(
        Path("src") / "core" / "config_agents.yaml"
    )
    pipeline_conspectius = ConspectiusPipeline(*configs)

    conspect = pipeline_conspectius.pipeline(litle_audio_file_path)

    if conspect is None:
        logger.warning("Лекция пуста на академический контент.")
        return

    logger.success(f"Финальный конспект сохранен по пути: {conspect}!")


if __name__ == "__main__":
    main()
