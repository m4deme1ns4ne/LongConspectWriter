from loguru import logger
from tqdm import tqdm
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
import json
from src.core.utils import ColoursForTqdm


class _AgentExtractor(BaseLlamaCppAgent):
    def __init__(self, session_dir: Path, **kwargs):
        self.session_dir = session_dir
        super().__init__(**kwargs)
        with open(self._app_config.scheme_output_path, "r", encoding="utf-8") as file:
            scheme_output = json.load(file)
        self.response_format = {"type": "json_object", "schema": scheme_output}

    def run(self, synthesizer_chunk: str = None) -> str:
        with tqdm(
            total=self._gen_config.max_tokens,
            desc="Экстракриция тем",
            unit="токен",
            colour=ColoursForTqdm.second_level,
            leave=False,
            position=1,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as token_pbar:
            response = self._generate(
                prompt=self._build_prompt(text=synthesizer_chunk),
                stream=True,
                token_pbar=token_pbar,
                response_format=self.response_format,
            )
        logger.debug(
            f"Экстракция чанка сгенерированного синтизером завершена, его длинна: {len(self.model.tokenize(response.encode('utf-8')))}"
        )

        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError as error:
            logger.error(
                f"LLM вернула сломанный JSON, экстракция пропущена. Ошибка: {error}"
            )
            response_dict = {"extracted_entities": []}

        _ = self._safe_result_out_line(
            output=response_dict,
            stage=self._app_config.name_stage_dir,
            file_name="out_filepath.jsonl",
            session_dir=self.session_dir,
            extension_file_writer="a",
        )

        return response_dict
