from loguru import logger
from tqdm import tqdm
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
import json
import ast
from src.core.utils import ColoursForTqdm


class AgentGrapher(BaseLlamaCppAgent):
    def __init__(self, init_config, gen_config, app_config, session_dir: Path):
        self.session_dir = session_dir
        super().__init__(init_config, gen_config, app_config)

    def run(self, path: Path = None, synthesizer_chunk: str = None) -> str:
        if path is not None:
            with open(path, "r", encoding="utf-8") as file:
                synthesizer_chunk = file.read()

        with open(self._app_config.scheme_output_path, "r", encoding="utf-8") as file:
            scheme_output = json.load(file)

        response_format = {"type": "json_object", "schema": scheme_output}

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
                prompt=self._build_prompt(text=synthesizer_chunk),
                stream=True,
                token_pbar=token_pbar,
                response_format=response_format,
            )

        response_dict = ast.literal_eval(response)

        out_filepath = self._safe_result_out_line(
            output_dict=response_dict,
            stage="05.5_extractor/",
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        logger.success(f"Глобальные заголовки сохранены: {out_filepath}")
        return out_filepath
