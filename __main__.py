import yaml
import argparse
from pathlib import Path

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
    PipelineSessionConfig,
    LocalClusterizerInitConfig,
    LocalClusterizerGenConfig,
    GlobalClusterizerInitConfig,
)
from src.core.utils import load_agent_bundle


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
            "synthesizer",
            "planner",
            "local_planner",
            "global_planner",
            "clustering",
            "local_clustering",
            "global_clustering",
            "convert_json_to_md",
            "grapher",
            "add_graph_in_conspect",
            "graph_planner",
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
        "--graphs_path",
        type=str,
        required=False,
        help="Путь к JSON-файлу с путями сгенерированных графиков",
    )
    parser.add_argument(
        "--continue_pipeline",
        type=str,
        required=False,
        help="Продолжить пайплайн с выбранного action и до конца",
    )
    args = parser.parse_args()

    args.output_file_path = "src/configs/config_pipeline.yaml"

    with open(args.output_file_path, "r", encoding="utf-8") as file:
        raw_pipeline_cfg = yaml.safe_load(file)
    pipeline_config = PipelineConfig(**raw_pipeline_cfg.get("app_config", {}))

    stt_bundle = load_agent_bundle(
        "src/configs/config-agents/stt/config_stt.yaml",
        STTInitConfig,
        STTGenConfig,
        AppSTTConfig,
    )
    synth_bundle = load_agent_bundle(
        "src/configs/config-agents/synthesizer/config_synthesizer.yaml",
        LLMInitConfig,
        LLMGenConfig,
        LLMAppConfig,
    )
    lp_bundle = load_agent_bundle(
        "src/configs/config-agents/local_planner/config_local_planner.yaml",
        LLMInitConfig,
        LLMGenConfig,
        LLMAppConfig,
    )
    gp_bundle = load_agent_bundle(
        "src/configs/config-agents/global_planner/config_global_planner.yaml",
        LLMInitConfig,
        LLMGenConfig,
        LLMAppConfig,
    )
    ex_bundle = load_agent_bundle(
        "src/configs/config-agents/extractor/config_extractor_planner.yaml",
        LLMInitConfig,
        LLMGenConfig,
        LLMAppConfig,
    )

    local_cluster_bundle = load_agent_bundle(
        "src/configs/config-clusterizer/config_local_clusterizer.yaml",
        LocalClusterizerInitConfig,
        LocalClusterizerGenConfig,
        None,
    )
    global_cluster_bundle = load_agent_bundle(
        "src/configs/config-clusterizer/config_global_clusterizer.yaml",
        GlobalClusterizerInitConfig,
        None,
        None,
    )

    grph_bundle = load_agent_bundle(
        "src/configs/config-agents/grapher/config_grapher.yaml",
        LLMInitConfig,
        LLMGenConfig,
        LLMAppConfig,
    )
    grph_pl_bundle = load_agent_bundle(
        "src/configs/config-agents/graph_planner/config_graph_planner.yaml",
        LLMInitConfig,
        LLMGenConfig,
        LLMAppConfig,
    )

    session_config = PipelineSessionConfig(
        pipeline=pipeline_config,
        stt=stt_bundle,
        synthesizer=synth_bundle,
        local_planner=lp_bundle,
        global_planner=gp_bundle,
        local_clusterizer=local_cluster_bundle,
        global_clusterizer=global_cluster_bundle,
        extractor=ex_bundle,
        grapher=grph_bundle,
        graph_planner=grph_pl_bundle,
    )

    pipeline = LongConspectWriterPipeline(session_config)

    if args.action == "global_clustering":
        global_plan_path = Path(args.global_plan_path)
        local_clusters_path = Path(args.local_clusters_path)

        output_path = pipeline._call_global_clustering(
            global_plan_path, local_clusters_path
        )

    elif args.action == "add_graph_in_conspect":
        conspect_md_path = Path(args.path_to_file)
        graphs_path = Path(args.graphs_path)

        output_path = pipeline.add_graph_in_conspect(
            graphs_path=graphs_path, conspect_md_path=conspect_md_path
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

        elif args.action == "convert_json_to_md":
            output_path = pipeline.convert_json_to_md(path_to_file)

        elif args.action == "grapher":
            output_path = pipeline._call_grapher(path_to_file)

        elif args.action == "graph_planner":
            output_path = pipeline._call_graph_planner(path_to_file)

    if output_path is not None:
        logger.info("Работа завершена.")
        return

    logger.critical("Переменная output_path пустая!")
    return


if __name__ == "__main__":
    main()
