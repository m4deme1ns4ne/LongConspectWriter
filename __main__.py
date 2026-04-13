from src.core.pipeline import ConspectiusPipeline
from loguru import logger
from dotenv import load_dotenv
import os
from pathlib import Path
import yaml
import argparse
from src.configs.ai_configs import (
    AppLLMConfig,
    AppSTTConfig,
    LLMGenConfig,
    LLMInitConfig,
    STTGenConfig,
    STTInitConfig,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_pipeline_configs(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    stt_init_config = STTInitConfig(**config["stt_init_config"])
    stt_gen_config = STTGenConfig(**config["stt_gen_config"])
    stt_app_config = AppSTTConfig(**config["stt_app_config"])

    drafter_init = LLMInitConfig(**config["drafter_init_config"])
    drafter_gen = LLMGenConfig(**config["drafter_gen_config"])
    drafter_app = AppLLMConfig(**config["drafter_app_config"])

    synth_init = LLMInitConfig(**config["synthesizer_init_config"])
    synth_gen = LLMGenConfig(**config["synthesizer_gen_config"])
    synth_app = AppLLMConfig(**config["synthesizer_app_config"])

    local_planner_init = LLMInitConfig(**config["local_planner_init_config"])
    local_planner_gen = LLMGenConfig(**config["local_planner_gen_config"])
    local_planner_app = AppLLMConfig(**config["local_planner_app_config"])

    global_planner_init = LLMInitConfig(**config["global_planner_init_config"])
    global_planner_gen = LLMGenConfig(**config["global_planner_gen_config"])
    global_planner_app = AppLLMConfig(**config["global_planner_app_config"])

    return (
        stt_init_config,
        stt_gen_config,
        stt_app_config,
        drafter_init,
        drafter_gen,
        drafter_app,
        synth_init,
        synth_gen,
        synth_app,
        local_planner_init,
        local_planner_gen,
        local_planner_app,
        global_planner_init,
        global_planner_gen,
        global_planner_app,
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="CLI для управления MAS LongConspectWriter."
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=[
            "all",
            "stt",
            "drafter",
            "synthesizer",
            "planner",
            "local_planner",
            "global_planner",
            "clustering",
            "local_clustering",
            "global_clustering",
        ],
        default="all",
        help="Тип операция с MAS",
    )
    parser.add_argument(
        "--path_to_file",
        type=str,
        required=False,
        help="Путь к входному файлу (аудио или текст)",
    )
    parser.add_argument(
        "--global_plan_path",
        type=str,
        required=False,
        help="Путь к входному файлу для глобального плана",
    )
    parser.add_argument(
        "--local_clusters_path",
        type=str,
        required=False,
        help="Путь к входному файлу для локальных кластеров",
    )
    args = parser.parse_args()
    action = args.action

    config = load_pipeline_configs(Path("src") / "configs" / "config.yaml")
    pipeline_conspectius = ConspectiusPipeline(*config)

    if action == "global_clustering":
        if args.global_plan_path is None:
            logger.critical("Аргумент --global_plan_path не передан при запуске.")
            return
        if args.local_clusters_path is None:
            logger.critical("Аргумент --local_clusters_path не передан при запуске.")
            return
        global_plan_path = Path(args.global_plan_path)
        local_clusters_path = Path(args.local_clusters_path)
        if not global_plan_path.is_file():
            logger.critical(
                f"Файла {global_plan_path.name} не существует. Указанный путь до файла: {global_plan_path}"
            )
            return

        if not local_clusters_path.is_file():
            logger.critical(
                f"Файла {local_clusters_path.name} не существует. Указанный путь до файла: {local_clusters_path}"
            )
            return

        output_path = pipeline_conspectius._call_global_clustering(
            global_plan_path, local_clusters_path
        )

    else:
        path_to_file = Path(args.path_to_file)
        if not path_to_file.is_file():
            logger.critical(f"Файл не существует. Путь до файла: {path_to_file}")
            return

        if action == "all":
            output_path = pipeline_conspectius.run(path_to_file)

        elif action == "stt":
            output_path = pipeline_conspectius._call_stt(path_to_file)

        elif action == "drafter":
            output_path = pipeline_conspectius._call_drafter(path_to_file)

        elif action == "synthesizer":
            output_path = pipeline_conspectius._call_synthesizer(path_to_file)

        elif action == "planner":
            output_path = pipeline_conspectius._call_planner(path_to_file)

        elif action == "local_planner":
            output_path = pipeline_conspectius._call_local_planner(path_to_file)

        elif action == "global_planner":
            output_path = pipeline_conspectius._call_global_planner(path_to_file)

        elif action == "local_clustering":
            output_path = pipeline_conspectius._call_local_clustering(path_to_file)

        elif action == "clustering":
            output_path = pipeline_conspectius._call_clustering(path_to_file)

    if output_path is not None:
        logger.info(f"Работа завершена.")
        return

    logger.critical(f"Переменная output_path пустая!")
    return


if __name__ == "__main__":
    main()
