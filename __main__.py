from src.pipline import ConspectiusPipline
from loguru import logger
from dotenv import load_dotenv
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    load_dotenv()
    full_audio_file_path = r"data\example-audio\Защита информации. Лекция 1..m4a"
    litle_audio_file_path = (
        r"data\example-audio\Защита информации. Лекция 1. (mp3cut.net).mp3"
    )
    pipline_conspectius = ConspectiusPipline()
    conspect = pipline_conspectius.pipline(full_audio_file_path)
    logger.success(f"Финальный конспект сохранен по пути: {conspect}!")


if __name__ == "__main__":
    main()
