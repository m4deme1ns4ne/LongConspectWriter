import os
from tqdm import tqdm
import time
import sys
from pathlib import Path
import json

from loguru import logger
from src.agents.base_agent import BaseTransformersAgent
from src.core.utils import TqdmTokenStreamer


class AgentSynthesizer(BaseTransformersAgent):
    def __init__(self, init_config, gen_config, app_config):
        super().__init__(init_config, gen_config, app_config)

    # def _build_prompt(self, **kwargs) -> str:
    #     user_prompt = self.user_template.format(**kwargs)
    #     full_prompt = (
    #         f"{self.system_prompt}\n"
    #         f"{user_prompt}\n"
    #         "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #     )

    #     return full_prompt

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            global_clusters = json.load(file)

        conspect = {topik: [] for topik, _ in global_clusters.items()}
        with tqdm(
            total=len(global_clusters),
            unit="кластер",
            desc="Кластеры",
            colour="green",
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for topik, clusters in global_clusters.items():
                split_clusters = " ".join(clusters)
                with tqdm(
                    total=self._gen_config.max_new_tokens,
                    desc="Генерация токенов",
                    unit="токен",
                    colour="blue",
                    leave=False,
                    position=1,
                    file=sys.stdout,
                    dynamic_ncols=True,
                ) as token_pbar:
                    streamer = TqdmTokenStreamer(token_pbar)
                    response = self._generate(
                        prompt=self._build_prompt(
                            text=split_clusters, cluster_topik=topik
                        ),
                        streamer=streamer,
                    )
                pbar.update(1)
                conspect[topik].append(response)

        logger.info("Генерация финального конспекта завершена.")

        timestamp = int(time.time())
        safe_model_name = self._init_config.pretrained_model_name_or_path.replace(
            "/", "_"
        )
        pure_draft_name = os.path.basename(path)
        out_filepath = os.path.join(
            self._app_config.output_dir,
            f"{safe_model_name}-{pure_draft_name}-{timestamp}.json",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as file:
            json.dump(conspect, file, ensure_ascii=False, indent=4)
        logger.success(f"Финальный конспект сохранен: {out_filepath}")
        return out_filepath
