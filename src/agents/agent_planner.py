from loguru import logger
from tqdm import tqdm
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
from src.core.utils import ColoursForTqdm
import json
import ast


class AgentLocalPlanner(BaseLlamaCppAgent):
    def __init__(self, session_dir: Path, **kwargs):
        self.session_dir = session_dir
        super().__init__(**kwargs)

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            local_clusters = json.load(file)

        local_clusters = list(local_clusters.values())

        final_drafts = []
        ignored_chunks = 0

        with tqdm(
            total=len(local_clusters),
            unit="чанк",
            desc="Локальные кластеры",
            colour=ColoursForTqdm.first_level,
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in local_clusters:
                with tqdm(
                    total=self._gen_config.max_tokens,
                    desc="Генерация токенов",
                    unit="токен",
                    colour=ColoursForTqdm.second_level,
                    leave=False,
                    position=1,
                    file=sys.stdout,
                    dynamic_ncols=True,
                ) as token_pbar:
                    response = self._generate(
                        prompt=self._build_prompt(text=chunk),
                        stream=True,
                        token_pbar=token_pbar,
                    )
                pbar.update(1)
                if "[NO_TOPICS]" in response:
                    ignored_chunks += 1
                    continue

                final_drafts.append(response)

        logger.info(
            f"Генерация локальных заголовков завершена. Кол-во локальных заголовков: {len(final_drafts)}. Кол-во пропущенный заголовков: {ignored_chunks}"
        )

        full_local_plan = "\nЗаголовок: ".join(final_drafts)
        response_dict = {"answer_agent": full_local_plan}
        out_filepath = self._safe_result_out_line(
            output=response_dict,
            stage=self._app_config.name_stage_dir,
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )
        return out_filepath


class AgentGlobalPlanner(BaseLlamaCppAgent):
    def __init__(self, session_dir: Path, **kwargs):
        self.session_dir = session_dir
        super().__init__(**kwargs)

    def run(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as file:
            local_plan = json.load(file)

        with open(self._app_config.scheme_output_path, "r", encoding="utf-8") as file:
            scheme_output = json.load(file)

        response_format = {"type": "json_object", "schema": scheme_output}

        local_plan = local_plan["answer_agent"]
        with tqdm(
            total=self._gen_config.max_tokens,
            desc="Генерация токенов",
            unit="токен",
            colour=ColoursForTqdm.first_level,
            leave=False,
            position=1,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as token_pbar:
            response = self._generate(
                prompt=self._build_prompt(text=local_plan),
                stream=True,
                token_pbar=token_pbar,
                response_format=response_format,
            )
        logger.info("Генерация глобальных заголовков завершена.")

        response_dict = ast.literal_eval(response)

        out_filepath = self._safe_result_out_line(
            output=response_dict,
            stage=self._app_config.name_stage_dir,
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.success(f"Глобальные заголовки сохранены: {out_filepath}")
        return out_filepath
