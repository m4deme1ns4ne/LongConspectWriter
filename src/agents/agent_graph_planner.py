"""Агент планирования графиков для Markdown-конспекта LongConspectWriter.

Агент просматривает синтезированный Markdown-конспект, спрашивает LLM, где
визуализации полезны, и вставляет плейсхолдеры графиков для этапа grapher.
"""

from loguru import logger
from tqdm import tqdm
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
import json
from src.core.utils import ColoursForTqdm, TextsSplitter, modify_retry
import difflib
from typing import Any


class AgentGraphPlanner(BaseLlamaCppAgent):
    """Добавляет плейсхолдеры графиков в Markdown-конспект."""

    def __init__(self, session_dir: Path, **kwargs: Any) -> None:
        """Инициализирует планирование графиков со схемой структурированного вывода.

        Args:
            session_dir (Path): Директория текущей сессии пайплайна.
            **kwargs (Any): Конфигурация LLM, передаваемая в ``BaseLlamaCppAgent``.

        Returns:
            None: Планировщик сохраняет сессию, модель и состояние формата ответа.
        """
        self.session_dir = session_dir
        super().__init__(**kwargs)
        with open(self._app_config.scheme_output_path, "r", encoding="utf-8") as file:
            scheme_output = json.load(file)
        self.response_format = {"type": "json_object", "schema": scheme_output}

    def _format_output(self, user_prompt: str) -> list[dict[str, str]]:
        """Форматирует prompt планирования графиков как одно пользовательское сообщение.

        Args:
            user_prompt (str): Пользовательский prompt, построенный из чанка Markdown-конспекта.

        Returns:
            list[dict[str, str]]: Chat prompt для llama.cpp.
        """
        combined_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        return [{"role": "user", "content": combined_prompt}]

    @modify_retry
    def create_graph_place_holder(
        self,
        conspect_chunk: str,
        conspect_theme: str | None,
        path_with_output: str | Path,
    ) -> dict[str, Any]:
        """Просит LLM проанализировать один чанк конспекта на возможности для графиков.

        Args:
            conspect_chunk (str): Markdown-чанк из синтезированного конспекта.
            conspect_theme (str | None): Тема лекции, используемая как дополнительный
                контекст для планирования графиков.
            path_with_output (str | Path): Путь JSONL, куда дописываются сырые ответы LLM
                для контроля.

        Returns:
            dict[str, Any]: Распарсенный ответ анализа графиков, обычно со списком
            ``analysis``.
        """
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
                    text=conspect_chunk,
                    conspect_theme=conspect_theme,
                    available_lib=self._app_config.available_lib,
                ),
                stream=True,
                token_pbar=pbar,
                response_format=self.response_format,
            )
        with open(path_with_output, "a", encoding="utf-8") as f:
            f.write(response)
        return json.loads(response)

    def normalizing_text(self, text: str) -> tuple[str, list[int]]:
        """Нормализует текст для нечеткого сопоставления цитат с сохранением индексов.

        Args:
            text (str): Текст конспекта или цитаты, участвующей в расстановке плейсхолдеров.

        Returns:
            tuple[str, list[int]]: Нормализованный текст и карта из нормализованных
            позиций обратно в исходные индексы текста.
        """
        junk_chars = set(" \t\n\r\\{}_^$.,:;-—()[]")
        chars = []
        index_map = []
        for i, char in enumerate(text):
            if char.isspace() or char in junk_chars:
                continue
            chars.append(char.lower())
            index_map.append(i)
        return "".join(chars), index_map

    @staticmethod
    def _collect_unsafe_ranges(text: str) -> list[tuple[int, int]]:
        """Возвращает список (start, end) диапазонов внутри парных структур Markdown/LaTeX."""
        ranges = []
        i = 0
        n = len(text)
        while i < n:
            if text[i : i + 3] == "```":
                end = text.find("```", i + 3)
                if end != -1:
                    ranges.append((i, end + 3))
                    i = end + 3
                else:
                    i += 3
            elif text[i : i + 2] == "$$":
                end = text.find("$$", i + 2)
                if end != -1:
                    ranges.append((i, end + 2))
                    i = end + 2
                else:
                    i += 2
            elif text[i] == "$":
                end = text.find("$", i + 1)
                if end != -1:
                    ranges.append((i, end + 1))
                    i = end + 1
                else:
                    i += 1
            elif text[i] == "`":
                end = text.find("`", i + 1)
                if end != -1:
                    ranges.append((i, end + 1))
                    i = end + 1
                else:
                    i += 1
            elif (
                text[i] == "<"
                and i + 1 < n
                and (text[i + 1].isalpha() or text[i + 1] == "/")
            ):
                end = text.find(">", i + 1)
                if end != -1:
                    ranges.append((i, end + 1))
                    i = end + 1
                else:
                    i += 1
            elif text[i] == "[":
                end_b = text.find("]", i + 1)
                if end_b != -1 and end_b + 1 < n and text[end_b + 1] == "(":
                    end_p = text.find(")", end_b + 2)
                    if end_p != -1:
                        ranges.append((i, end_p + 1))
                        i = end_p + 1
                        continue
                i += 1
            else:
                i += 1
        return ranges

    @staticmethod
    def _find_safe_insert_pos(text: str, start_idx: int) -> int:
        """Сдвигает start_idx вперёд, пока он не окажется вне любой парной структуры."""
        unsafe = AgentGraphPlanner._collect_unsafe_ranges(text)
        pos = start_idx
        changed = True
        while changed:
            changed = False
            for a, b in unsafe:
                if a < pos < b:
                    pos = b
                    changed = True
                    break
        return pos

    def _apply_graphs_to_markdown(
        self, conspect_md: str, analysis_results: list[dict[str, Any]]
    ) -> str:
        """Вставляет плейсхолдеры графиков после найденных цитат в Markdown.

        Args:
            conspect_md (str): Полный текст Markdown-конспекта.
            analysis_results (list[dict[str, Any]]): Ответы graph planner для
                каждого чанка конспекта.

        Returns:
            str: Markdown-текст со вставленными тегами плейсхолдеров графиков.
        """
        for analys in analysis_results:
            for item in analys.get("analysis", []):
                if str(item.get("decision")).lower() == "true":
                    quote = item["quote"].strip()
                    if len(quote) < 20:
                        logger.warning(f"Слишком короткая цитата, пропускаем: {quote}")
                        continue
                    graph_type = item.get("type", "Визуализация")
                    title = item.get("title", "График")
                    mock_data = item.get("mock_data", "")
                    task = item.get("task", "")

                    tag = f"[GRAPH_TYPE: {graph_type} | GRAPH_TITLE: {title} | MOCK_DATA: {mock_data} | TASK: {task}]"

                    normalized_conspect_md, real_imap = self.normalizing_text(
                        conspect_md
                    )
                    normalized_quote, _ = self.normalizing_text(quote)

                    matcher = difflib.SequenceMatcher(
                        None, normalized_conspect_md, normalized_quote, autojunk=False
                    )
                    match = matcher.find_longest_match(
                        0, len(normalized_conspect_md), 0, len(normalized_quote)
                    )

                    min_match_len = min(
                        40, len(normalized_quote) * 0.7
                    )  # Потом убрать и добавить в конфиг

                    if match.size >= min_match_len:
                        last_norm_idx = match.a + match.size - 1
                        last_real_idx = real_imap[last_norm_idx]
                        insert_idx = self._find_safe_insert_pos(
                            conspect_md, last_real_idx + 1
                        )

                        conspect_md = (
                            conspect_md[:insert_idx]
                            + f"\n\n{tag}\n\n"
                            + conspect_md[insert_idx:]
                        )
                    else:
                        logger.warning(
                            f"Цитата не найдена в конспекте (difflib). Максимальное непрерывное совпадение: {match.size}, требуется минимум: {int(min_match_len)}. Текст: {quote}"
                        )
        return conspect_md

    def run(
        self, conspect_md_path: str | Path, conspect_theme: str | None = None
    ) -> Path:
        """Планирует плейсхолдеры графиков для синтезированного Markdown-конспекта.

        Args:
            conspect_md_path (str | Path): Путь к Markdown-конспекту.
            conspect_theme (str | None): Опциональная тема лекции, передаваемая в
                prompt планирования графиков.

        Returns:
            Path: Путь к Markdown с тегами плейсхолдеров графиков.
        """
        with open(conspect_md_path, "r", encoding="utf-8") as file:
            conspect_md = file.read()
        conspect_chunks = TextsSplitter.split_text_to_tokens(
            text=conspect_md,
            tokenizer=self.model.tokenize,
            chunk_size=int(
                self._gen_config.max_tokens * 0.5
            ),  # Потом убрать и добавить в конфиг
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
            for chunk in conspect_chunks:
                path_with_output = self._get_output_file_path(
                    session_dir=self.session_dir,
                    stage=self._app_config.name_stage_dir,
                    file_name=f"out_filepath.jsonl",
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
                if str(item.get("decision")).lower() == "true":
                    total_graphs += 1

        logger.debug(f"Найдено плейсхолдеров для графиков: {total_graphs}")

        conspect_md = self._apply_graphs_to_markdown(
            conspect_md=conspect_md, analysis_results=analysis_results
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
