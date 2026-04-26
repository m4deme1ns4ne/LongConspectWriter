from loguru import logger
from tqdm import tqdm
import sys
from src.core.base import BaseLlamaCppAgent
from pathlib import Path
from src.core.utils import ColoursForTqdm
import re
import subprocess
import os


class AgentGrapher(BaseLlamaCppAgent):
    def __init__(self, session_dir: Path, **kwargs):
        self.session_dir = session_dir
        super().__init__(**kwargs)

    def getting_graphs_from_conspect(self, conspect: str) -> list[tuple[int, int, str]]:
        """
        Ищет все теги [GRAPH: ...] с учетом вложенности скобок.
        Возвращает список кортежей (индекс_начала, индекс_конца, сам_текст_тега).
        """
        graphs = []
        char_open_count = 0
        idx_start = 0

        for i, char in enumerate(conspect):
            if char == "[":
                if char_open_count == 0:
                    if conspect[i : i + 7] == "[GRAPH:":
                        char_open_count = 1
                        idx_start = i
                else:
                    char_open_count += 1
            elif char == "]":
                if char_open_count > 0:
                    char_open_count -= 1
                    if char_open_count == 0:
                        left_bound = max(0, idx_start - 200)
                        right_bound = min(len(conspect), i + 201)
                        graphs.append((idx_start, i, conspect[left_bound:right_bound]))
        return graphs

    def _generate_graph_code(self, description: str, target_image_path: Path) -> str:
        """
        Изолированный метод для обращения к LLM.
        """
        with tqdm(
            total=self._gen_config.max_tokens,
            desc=f"Генерация кода для {target_image_path.name}",
            unit="токен",
            colour=ColoursForTqdm.fourth_level,
            leave=False,
            position=2,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as chunk_token_pbar:
            response = self._generate(
                prompt=self._build_prompt(
                    text=description, target_path=str(target_image_path.absolute())
                ),
                stream=True,
                token_pbar=chunk_token_pbar,
            )
        return response

    def _code_call(
        self, code: str, expected_image_path: Path, script_path: Path
    ) -> bool:
        """
        Сохраняет код в файл и выполняет его в подпроцессе.
        Скрипты больше не удаляются для возможности дебага.
        """
        match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
        if not match:
            logger.error(f"LLM не вернула блок кода для {expected_image_path.name}")
            return False

        python_script = match.group(1)

        # Аппаратная инъекция настроек кириллицы
        cyrillic_setup = (
            "import matplotlib.pyplot as plt\n"
            "plt.rcParams['font.family'] = 'Arial'\n" # Или 'DejaVu Sans', если Arial нет в системе
            "plt.rcParams['axes.unicode_minus'] = False\n\n"
        )
        python_script = cyrillic_setup + python_script

        # Сохраняем скрипт для истории и валидации
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(python_script)

        logger.debug(f"Запуск скрипта генерации: {script_path}")

        try:
            # Копируем текущие переменные окружения ОС
            env = os.environ.copy()
            # Принудительно отключаем GUI в matplotlib
            env["MPLBACKEND"] = "Agg"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=15.0,
                env=env,
            )

            if result.returncode != 0:
                logger.error(
                    f"Ошибка рендера {expected_image_path.name}: {result.stderr}"
                )
                return False

            if expected_image_path.exists():
                return True
            else:
                logger.error(
                    f"Скрипт выполнился, но не создал {expected_image_path.name}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(
                f"Превышен таймаут рендера (15с) для {expected_image_path.name}. Процесс убит."
            )
            return False

    def run(self, path: Path = None) -> str:
        with open(path, "r", encoding="utf-8") as file:
            conspect = file.read()

        graphs_data = self.getting_graphs_from_conspect(conspect)

        # Инициализируем словарь для маппинга: { "оригинальный_тег": "относительный_путь" }
        graphs_mapping = {}

        if not graphs_data:
            logger.info("Плейсхолдеров [GRAPH: ...] не найдено. Пропуск.")
            # Возвращаем пустой JSON
            return str(
                self._safe_result_out_line(
                    output_dict=graphs_mapping,
                    stage="08_grapher/",
                    file_name="graphs_mapping.json",
                    session_dir=self.session_dir,
                    extension="json",
                )
            )

        logger.info(f"Найдено графиков для рендера: {len(graphs_data)}")

        # Прямая итерация, так как мы больше не делаем replace в строке на лету
        for i, (idx_start, idx_end, description) in tqdm(
            enumerate(graphs_data),
            total=len(graphs_data),
            desc="Обработка графиков",
            colour=ColoursForTqdm.first_level,
            position=1,
        ):
            # Получаем оригинальный тег (ключ для JSON)
            original_tag = conspect[idx_start : idx_end + 1]

            target_image_path = self._get_output_file_path(
                session_dir=self.session_dir,
                stage="08_grapher/assets",
                file_name=f"{i}_graph.png",
            )

            # Сохраняем код в папку scripts, а не temp
            script_path = self._get_output_file_path(
                session_dir=self.session_dir,
                stage="08_grapher/scripts",
                file_name=f"{i}_script.py",
            )

            code_response = self._generate_graph_code(description, target_image_path)
            is_success = self._code_call(code_response, target_image_path, script_path)

            if is_success:
                graphs_mapping[original_tag] = {
                    "status": "success",
                    "path": f"assets/{target_image_path.name}",
                }
            else:
                graphs_mapping[original_tag] = {"status": "error", "path": None}

        # Возвращаем JSON с маппингом для пайплайна
        out_filepath = self._safe_result_out_line(
            output_dict=graphs_mapping,
            stage="08_grapher/",
            file_name="graphs_mapping.json",
            session_dir=self.session_dir,
            extension="json",
        )

        logger.success(f"Маппинг графиков сохранен: {out_filepath}")
        return str(out_filepath)
