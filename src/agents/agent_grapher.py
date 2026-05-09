"""Агент рендера графиков для LongConspectWriter.

Агент превращает плейсхолдеры графиков от graph planner в Python-скрипты,
запускает их в ограниченном подпроцессе и сохраняет маппинг для финальной
сборки Markdown.
"""

from loguru import logger
from tqdm import tqdm
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
from src.core.utils import ColoursForTqdm
import re
import subprocess
import os
from typing import Any, Callable


class AgentGrapher(BaseLlamaCppAgent):
    """Генерирует визуальные ассеты графиков для Markdown-плейсхолдеров."""

    def __init__(
        self,
        session_dir: Path,
        getting_graphs_from_conspect_func: Callable[
            [Any, str], list[tuple[int, int, str]]
        ],
        **kwargs: Any,
    ) -> None:
        """Инициализирует рендеринг графиков с callback-функцией для извлечения плейсхолдеров.

        Args:
            session_dir (Path): Директория текущей сессии пайплайна.
            getting_graphs_from_conspect_func (Callable[[Any, str],
                list[tuple[int, int, str]]]): Callback-функция из пайплайна,
                которая находит теги плейсхолдеров графиков в конспекте.
            **kwargs (Any): Конфигурация LLM, передаваемая в ``BaseLlamaCppAgent``.

        Returns:
            None: Grapher сохраняет сессию, callback и состояние модели.
        """
        self.session_dir = session_dir
        self.getting_graphs_from_conspect = getting_graphs_from_conspect_func
        super().__init__(**kwargs)

    def _generate_graph_code(
        self, description: str, target_image_path: Path, error: str, bad_code: str
    ) -> str:
        """Генерирует Python-код для отрисовки одного плейсхолдера графика.

        Args:
            description (str): Контекст плейсхолдера и задача визуализации.
            target_image_path (Path): Ожидаемый путь к изображению, которое должен создать скрипт.
            error (str): Предыдущая ошибка отрисовки, передаваемая в повторные запросы.
            bad_code (str): Предыдущий неудачный код, передаваемый в повторные запросы.

        Returns:
            str: Сырой ответ LLM, который должен содержать блок Python-кода.
        """
        with tqdm(
            total=self._gen_config.max_tokens,
            desc=f"Генерация кода для {target_image_path.name[:10]}...",
            unit="токен",
            colour=ColoursForTqdm.fourth_level,
            leave=False,
            position=2,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as chunk_token_pbar:
            response = self._generate(
                prompt=self._build_prompt(
                    text=description,
                    target_path=str(target_image_path.absolute()),
                    error=str(error),
                    bad_code=str(bad_code),
                    available_lib=self._app_config.available_lib,
                ),
                stream=True,
                token_pbar=chunk_token_pbar,
            )
        return response

    def _code_call(
        self, code: str, expected_image_path: Path, script_path: Path
    ) -> bool | tuple[bool, str]:
        """Сохраняет сгенерированный код и выполняет его в подпроцессе.

        Args:
            code (str): Сырой ответ LLM, содержащий блок Python-кода.
            expected_image_path (Path): Путь к изображению, которое должен создать
                сгенерированный скрипт.
            script_path (Path): Путь к файлу, в который сохраняется сгенерированный скрипт.

        Returns:
            bool | tuple[bool, str]: Результат исходной ветки или флаг успешности вместе
            с диагностическим текстом в формате stderr.
        """
        match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if not match:
            logger.error(f"LLM не вернула блок кода для {expected_image_path.name}")
            return False

        python_script = match.group(1)

        cyrillic_setup = (
            "import matplotlib.pyplot as plt\n"
            "plt.rcParams['font.family'] = 'Arial'\n"
            "plt.rcParams['axes.unicode_minus'] = False\n\n"
        )
        python_script = cyrillic_setup + python_script

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(python_script)

        logger.debug(f"Запуск скрипта генерации: {script_path}")

        try:
            safe_keys = {
                "PATH",
                "PYTHONPATH",
                "PYTHONHOME",
                "SYSTEMROOT",
                "USERPROFILE",
                "TMP",
                "TEMP",
            }
            isolated_env = {k: v for k, v in os.environ.items() if k in safe_keys}

            isolated_env["MPLBACKEND"] = "Agg"

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=15.0,
                env=isolated_env,
            )

            if result.returncode != 0:
                logger.error(
                    f"Ошибка рендера {expected_image_path.name}: {result.stderr}"
                )
                return False, result.stderr

            if expected_image_path.exists():
                return True, ""
            else:
                logger.error(
                    f"Скрипт выполнился, но не создал {expected_image_path.name}"
                )
                return (
                    False,
                    "Скрипт завершился успешно, но файл картинки не был создан.",
                )

        except subprocess.TimeoutExpired:
            logger.error(
                f"Превышен таймаут рендера (15с) для {expected_image_path.name}. Процесс убит."
            )
            return False, "Превышен лимит времени выполнения (15 секунд)."

    def re_try(
        self, description: str, target_image_path: Path, script_path: Path
    ) -> bool:
        """Повторяет генерацию и рендеринг кода графика с изменённой температурой.

        Args:
            description (str): Контекст плейсхолдера и задача визуализации.
            target_image_path (Path): Ожидаемый путь к выходному изображению.
            script_path (Path): Путь, по которому записывается сгенерированный Python-код.

        Returns:
            bool: Было ли изображение графика успешно сгенерировано.
        """
        is_success = False
        bad_code = self._app_config.bad_code
        error_message = self._app_config.error_massage
        original_temp = self._gen_config.temperature

        for attempt in range(self._app_config.re_try_count):
            self._gen_config.temperature = original_temp + (
                attempt * self._app_config.step_temperature
            )

            if attempt > 0:
                logger.warning(
                    f"Ретрай {attempt}. Температура повышена до {self._gen_config.temperature}"
                )

            code_response = self._generate_graph_code(
                description=description,
                target_image_path=target_image_path,
                error=error_message,
                bad_code=bad_code,
            )

            is_success, stderr_text = self._code_call(
                code_response, target_image_path, script_path
            )

            if not is_success:
                bad_code = code_response
                error_message = (
                    f"ВНИМАНИЕ! Твой код упал с ошибкой:\n{stderr_text}\n"
                    f"СТРОГО ЗАПРЕЩЕНО ВОЗВРАЩАТЬ ТОТ ЖЕ САМЫЙ КОД! "
                    f"Если ошибка IndexError или ValueError, увеличь размерность mock-массивов (например, добавь элементы в массив m) "
                    f"или проверь границы циклов range()."
                )
                continue
            else:
                break

        self._gen_config.temperature = original_temp
        return is_success

    def run(self, path: str | Path | None = None) -> str:
        """Генерирует все изображения графиков, запрошенные плейсхолдерами в Markdown.

        Args:
            path (str | Path | None): Путь к Markdown с плейсхолдерами графиков.

        Returns:
            str: Путь к JSON, который связывает плейсхолдеры со сгенерированными изображениями.
        """
        with open(path, "r", encoding="utf-8") as file:
            conspect = file.read()

        graphs_data = self.getting_graphs_from_conspect(self, conspect=conspect)
        # Инициализируем словарь для маппинга: { "оригинальный_тег": "относительный_путь" }
        graphs_mapping = {}

        if not graphs_data:
            logger.info("Плейсхолдеров [GRAPH_TYPE: ...] не найдено. Пропуск.")
            # Возвращаем пустой JSON
            return str(
                self._safe_result_out_line(
                    output=graphs_mapping,
                    stage=self._app_config.name_stage_dir,
                    file_name="graphs_mapping.json",
                    session_dir=self.session_dir,
                    extension="json",
                )
            )

        logger.info(f"Найдено графиков для рендера: {len(graphs_data)}")

        for i, (idx_start, idx_end, description) in tqdm(
            enumerate(graphs_data),
            total=len(graphs_data),
            desc="Обработка графиков",
            colour=ColoursForTqdm.first_level,
            position=1,
        ):
            original_tag = conspect[idx_start : idx_end + 1]

            title_match = re.search(r"GRAPH_TITLE:\s*(.*?)(?:\||\])", original_tag)
            raw_title = title_match.group(1).strip() if title_match else "Иллюстрация"
            safe_title = re.sub(r"[^\w\-]", "_", raw_title)

            target_image_path = self._get_output_file_path(
                session_dir=self.session_dir,
                stage=f"{self._app_config.name_stage_dir}/assets",
                file_name=f"{i}___{safe_title}.png",
            )

            # Сохраняем код в папку scripts, а не temp
            script_path = self._get_output_file_path(
                session_dir=self.session_dir,
                stage=f"{self._app_config.name_stage_dir}/scripts",
                file_name=f"{i}___{safe_title}.py",
            )

            is_success = self.re_try(
                description=description,
                target_image_path=target_image_path,
                script_path=script_path,
            )

            if is_success:
                graphs_mapping[original_tag] = {
                    "status": "success",
                    "path": f"assets/{target_image_path.name}",
                    "name_graph": target_image_path.name,
                }
            else:
                graphs_mapping[original_tag] = {"status": "error", "path": None}

        # Возвращаем JSON с маппингом для пайплайна
        out_filepath = self._safe_result_out_line(
            output=graphs_mapping,
            stage=self._app_config.name_stage_dir,
            file_name="graphs_mapping.json",
            session_dir=self.session_dir,
            extension="json",
        )

        logger.success(f"Маппинг графиков сохранен: {out_filepath}")
        return str(out_filepath)
