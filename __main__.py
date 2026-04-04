from src.pipeline import ConspectiusPipeline
from loguru import logger
from dotenv import load_dotenv
import os

"""
1) сделать более умный bad_words с алгоритмом https://claude.ai/chat/4fe04762-3755-444f-92f2-d6e08536fb27

2) разобраться с т2
"""

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main() -> None:
    load_dotenv()
    full_audio_file_path = r"data\example-audio\Лекция 1. Вещественные числа. Часть 1.mp3"
    # litle_audio_file_path = (
    #     r"data\example-audio\Защита информации. Лекция 1. (mp3cut.net).mp3"
    # )
    pipeline_conspectius = ConspectiusPipeline()
    conspect = pipeline_conspectius.pipeline(full_audio_file_path)
    if conspect is None:
        logger.warning("Лекция пуста на академический контент.")
        return
    logger.success(f"Финальный конспект сохранен по пути: {conspect}!")


if __name__ == "__main__":
    main()
