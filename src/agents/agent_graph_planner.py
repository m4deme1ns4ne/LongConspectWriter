from loguru import logger
from tqdm import tqdm
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
import json
from src.core.utils import ColoursForTqdm, TextsSplitter, modify_retry
import difflib


class AgentGraphPlanner(BaseLlamaCppAgent):
    def __init__(self, session_dir: Path, **kwargs):
        self.session_dir = session_dir
        super().__init__(**kwargs)
        with open(self._app_config.scheme_output_path, "r", encoding="utf-8") as file:
            scheme_output = json.load(file)
        self.response_format = {"type": "json_object", "schema": scheme_output}

    def _format_output(self, user_prompt):
        combined_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        return [{"role": "user", "content": combined_prompt}]

    @modify_retry
    def create_graph_place_holder(
        self, conspect_chunk: str, conspect_theme: str, path_with_output: str
    ) -> dict:
        with tqdm(
            total=self._gen_config.max_tokens,
            desc="Генерация токенов для чанка",
            unit="токен",
            colour=ColoursForTqdm.fourth_level,
            leave=False,
            position=2,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            response = self._generate(
                prompt=self._build_prompt(
                    text=conspect_chunk, conspect_theme=conspect_theme
                ),
                stream=True,
                token_pbar=pbar,
                response_format=self.response_format,
            )
        with open(path_with_output, "w", encoding="utf-8") as f:
            f.write(response)
        return json.loads(response)

    def run(self, conspect_md_path: str, conspect_theme: str = None) -> str:
        with open(conspect_md_path, "r", encoding="utf-8") as file:
            conspect_md = file.read()
        conspect_chunks = TextsSplitter.split_text_to_tokens(
            text=conspect_md,
            tokenizer=self.model.tokenize,
            chunk_size=int(self._gen_config.max_tokens * 0.5),
            chunk_overlap=int(self._gen_config.max_tokens * 0.2),
        )
        analysis_results = []
        with tqdm(
            total=len(conspect_chunks),
            desc="Анализ чанков конспекта",
            unit="чанк",
            colour=ColoursForTqdm.second_level,
            leave=False,
            position=1,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            for i, chunk in enumerate(conspect_chunks):
                path_with_output = self._get_output_file_path(
                    session_dir=self.session_dir,
                    stage=self._app_config.name_stage_dir,
                    file_name=f"{i}_script.jsonl",
                )
                try:
                    response_dict = self.create_graph_place_holder(
                        conspect_chunk=chunk,
                        conspect_theme=conspect_theme,
                        path_with_output=path_with_output,
                    )
                except json.JSONDecodeError as error:
                    logger.error(
                        f"LLM вернула сломанный JSON после всех ретраев: {error}"
                    )
                    response_dict = {"analysis": []}
                analysis_results.append(response_dict)
                pbar.update(1)

        total_graphs = 0
        for analys in analysis_results:
            for item in analys.get("analysis", []):
                if str(item.get("decision")).lower() == "true" and item.get(
                    "placeholder"
                ):
                    total_graphs += 1

        logger.debug(f"Найдено плейсхолдеров для графиков: {total_graphs}")

        for analys in analysis_results:
            for item in analys.get("analysis", []):
                if str(item.get("decision")).lower() == "true" and item.get(
                    "placeholder"
                ):
                    quote = item["quote"].strip()
                    tag = item["placeholder"]

                    if len(quote) < 20:
                        logger.warning(f"Слишком короткая цитата, пропускаем: {quote}")
                        continue

                    matcher = difflib.SequenceMatcher(None, quote, conspect_md)
                    match = matcher.find_longest_match(
                        0, len(quote), 0, len(conspect_md)
                    )

                    min_match_len = min(40, len(quote) * 0.7)

                    if match.size >= min_match_len:
                        insert_index = match.b + match.size

                        conspect_md = (
                            conspect_md[:insert_index]
                            + f"\n\n{tag}"
                            + conspect_md[insert_index:]
                        )
                    else:
                        logger.warning(
                            f"Цитата не найдена в конспекте (difflib). Максимальное непрерывное совпадение: {match.size}, требуется минимум: {int(min_match_len)}. Текст: {quote}"
                        )

        out_filepath = self._safe_result_out_line(
            output=conspect_md,
            stage=self._app_config.name_stage_dir,
            file_name="out_filepath.md",
            session_dir=self.session_dir,
            extension_file_writer="w",
            extension="md",
        )

        return out_filepath
