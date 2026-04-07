from src.core.pipeline import ConspectiusPipeline
from loguru import logger
from dotenv import load_dotenv
import os
from pathlib import Path
from src.core.utils import load_pipeline_configs
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="CLI для управления MAS LongConspectWriter.")
    parser.add_argument(
            "--action", 
            type=str, 
            choices=["all", "stt", "drafter", "synthesizer"], 
            default="all", 
            help="Тип операция с MAS"
        )
    parser.add_argument(
        "--path_to_file", 
        type=str, 
        required=True, 
        help="Путь к входному файлу (аудио или текст)"
    )
    args = parser.parse_args()
    action = args.action
    path_to_file = Path(args.path_to_file)

    if not path_to_file.is_file():
        logger.critical(f"Файл не существует. Путь до файла: {path_to_file}")

    config_agents = load_pipeline_configs(Path("src") / "configs" / "config_agents.yaml")
    pipeline_conspectius = ConspectiusPipeline(*config_agents)

    if action == "all":
        output_path = pipeline_conspectius.pipeline(path_to_file)
        if output_path is not None:
            logger.success(f"Финальный конспект сохранен по пути: {output_path}!")
            return
    
    elif action == "stt":
        output_path = pipeline_conspectius._call_stt(path_to_file)

    elif action == "drafter":
        output_path = pipeline_conspectius._call_drafter(path_to_file)

    elif action == "synthesizer":
        output_path = pipeline_conspectius._call_synthesizer(path_to_file)

    if output_path is not None:
        logger.success(f"Ответ Агента {action} сохранены по пути: {output_path}!")
        return


if __name__ == "__main__":
    main()
