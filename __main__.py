import os
import yaml
import argparse
from pathlib import Path
# import warnings

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from src.core.pipeline import LongConspectWriterPipeline
from src.configs.configs import (
    AppSTTConfig,
    STTGenConfig,
    STTInitConfig,
    LLMInitConfig,
    LLMGenConfig,
    LLMAppConfig,
    PipelineConfig,
)

# # Оставляем только нужные системные фиксы, мусор от transformers удален
# warnings.filterwarnings("ignore", category=UserWarning)
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_configs(
    yaml_path,
    cls_init_config=None,
    cls_gen_config=None,
    cls_app_config=None,
):
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if cls_init_config is None:
        init_config = None
    else:
        init_config = cls_init_config(**config.get("init_config", {}))
    if cls_gen_config is None:
        gen_config = None
    else:
        gen_config = cls_gen_config(**config.get("gen_config", {}))
    if cls_app_config is None:
        app_config = None
    else:
        app_config = cls_app_config(**config.get("app_config", {}))

    return init_config, gen_config, app_config


def main() -> None:
    load_dotenv()
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    parser = argparse.ArgumentParser(
        description="CLI для управления LongConspectWriter."
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
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Путь к конфигу",
    )
    args = parser.parse_args()

    args.output_file_path = "src/configs/config_pipeline.yaml"
    _, _, pipeline_config = load_configs(
        args.output_file_path, cls_app_config=PipelineConfig
    )

    if args.config_path is None:
        # 1. Загрузка STT
        stt_init, stt_gen, stt_app = load_configs(
            "src/configs/config-agents/stt/config_stt.yaml",
            STTInitConfig,
            STTGenConfig,
            AppSTTConfig,
        )

        # 2. Загрузка Drafter
        drafter_init, drafter_gen, drafter_app = load_configs(
            "src/configs/config-agents/drafter/config_drafter.yaml",
            LLMInitConfig,
            LLMGenConfig,
            LLMAppConfig,
        )

        # 3. Загрузка Synthesizer
        synth_init, synth_gen, synth_app = load_configs(
            "src/configs/config-agents/synthesizer/config_synthesizer.yaml",
            LLMInitConfig,
            LLMGenConfig,
            LLMAppConfig,
        )

        # 4. Загрузка Local Planner
        lp_init, lp_gen, lp_app = load_configs(
            "src/configs/config-agents/local_planner/config_local_planner.yaml",
            LLMInitConfig,
            LLMGenConfig,
            LLMAppConfig,
        )

        # 5. Загрузка Global Planner
        gp_init, gp_gen, gp_app = load_configs(
            "src/configs/config-agents/global_planner/config_global_planner.yaml",
            LLMInitConfig,
            LLMGenConfig,
            LLMAppConfig,
        )
        pipeline = LongConspectWriterPipeline(
            pipeline_config,
            stt_init,
            stt_gen,
            stt_app,
            drafter_init,
            drafter_gen,
            drafter_app,
            synth_init,
            synth_gen,
            synth_app,
            lp_init,
            lp_gen,
            lp_app,
            gp_init,
            gp_gen,
            gp_app,
        )
    else:
        logger.warning("Архитектура с единым конфигом пока не реализована.")
        return

    if args.action == "global_clustering":
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

        output_path = pipeline._call_global_clustering(
            global_plan_path, local_clusters_path
        )

    else:
        path_to_file = Path(args.path_to_file)
        if not path_to_file.is_file():
            logger.critical(f"Файл не существует. Путь до файла: {path_to_file}")
            return

        if args.action == "all":
            output_path = pipeline.run(path_to_file)

        elif args.action == "stt":
            output_path = pipeline._call_stt(path_to_file)

        elif args.action == "drafter":
            output_path = pipeline._call_drafter(path_to_file)

        elif args.action == "synthesizer":
            output_path = pipeline._call_synthesizer(path_to_file)

        elif args.action == "planner":
            output_path = pipeline._call_planner(path_to_file)

        elif args.action == "local_planner":
            output_path = pipeline._call_local_planner(path_to_file)

        elif args.action == "global_planner":
            output_path = pipeline._call_global_planner(path_to_file)

        elif args.action == "local_clustering":
            output_path = pipeline._call_local_clustering(path_to_file)

        elif args.action == "clustering":
            output_path = pipeline._call_clustering(path_to_file)

    if output_path is not None:
        logger.info("Работа завершена.")
        return

    logger.critical("Переменная output_path пустая!")
    return


if __name__ == "__main__":
    main()
