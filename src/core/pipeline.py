"""Оркестрация последовательного пайплайна LongConspectWriter.

Модуль связывает STT, кластеризацию, планирование, синтез, планирование
графиков, рендер графиков и финальную сборку Markdown в релизный пайплайн.
Каждый метод сохраняет существующий контракт передачи путей между этапами.
"""

import os
from src.core.base import BasePipeline
from src.core.utils import check_path_is
from loguru import logger
import multiprocessing
from src.configs.configs import PipelineSessionConfig
import json
import shutil
from pathlib import Path


class LongConspectWriterPipeline(BasePipeline):
    """Координирует все этапы LongConspectWriter в строгом порядке."""

    def __init__(self, session_config: PipelineSessionConfig) -> None:
        """Инициализирует запуск пайплайна и создает директорию сессии.

        Args:
            session_config (PipelineSessionConfig): Fully loaded configuration
                for every LongConspectWriter stage.

        Returns:
            None: Пайплайн сохраняет конфиг и состояние активной сессии.

        Raises:
            OSError: Если выходную директорию сессии невозможно создать.
        """
        self.config = session_config
        self.pipeline_config = self.config.pipeline
        self.__post_init__()

    # @check_path_is
    # def _call_stt(self, path: Path) -> Path | None:
    #     with FasterWhisper(
    #         self.stt_init_config, self.stt_gen_config, self.stt_app_config
    #     ) as model:
    #         result_path = model.run(audio_file_path=path)

    #     return result_path

    def _run_stt_process(
        self, path: str | os.PathLike, result_queue: multiprocessing.Queue
    ) -> None:
        """Запускает STT в изолированном процессе и публикует путь результата.

        Args:
            path (str | os.PathLike): Source lecture audio/video path.
            result_queue (multiprocessing.Queue): Queue used to return status
                и путь транскрипта в родительский процесс пайплайна.

        Returns:
            None: Метод записывает словарь статуса в ``result_queue``.

        Raises:
            Exception: Обрабатывается внутри и сериализуется в очередь.
        """
        try:
            from src.core.stt import FasterWhisper

            faster_whisper = FasterWhisper(
                session_dir=self.actual_session_dir,
                init_config=self.config.stt.init_config,
                gen_config=self.config.stt.gen_config,
                app_config=self.config.stt.app_config,
                lecture_theme=self.pipeline_config.lecture_theme,
            )
            transcript_path = faster_whisper.run(audio_file_path=path)
            result_queue.put({"status": "success", "path": transcript_path})
        except Exception as e:
            result_queue.put({"status": "error", "error": str(e)})

    def _call_stt(self, path: str | os.PathLike) -> str:
        """Запускает этап транскрибации FasterWhisper.

        Args:
            path (str | os.PathLike): Путь к исходному аудио или видео лекции.

        Returns:
            str: Путь к сырому артефакту транскрипта, созданному STT.

        Raises:
            RuntimeError: Если изолированный STT-процесс сообщает об ошибке или завершается
                без возврата результата.
        """
        logger.info("Запуск STT агента в изолированном процессе...")
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_stt_process, args=(path, result_queue)
        )
        process.start()
        process.join()
        if not result_queue.empty():
            result = result_queue.get()
            if result["status"] == "success":
                return result["path"]
            else:
                raise RuntimeError(
                    f"Ошибка транскрибации в фоновом процессе: {result['error']}"
                )
        else:
            raise RuntimeError(
                "Процесс STT завершился, но не вернул результат. Возможно, произошло жесткое падение CTranslate2."
            )

    @check_path_is
    def _call_local_clustering(self, path: str | os.PathLike) -> str | os.PathLike:
        """Запускает локальную хронологическую кластеризацию по артефакту транскрипта.

        Args:
            path (str | os.PathLike): Путь к артефакту транскрипта.

        Returns:
            str | os.PathLike: Путь к локальным кластерам для этапов планирования.

        Raises:
            Exception: Пробрасывает ошибки локальной кластеризации.
        """
        from src.core.clustering import SemanticLocalClusterizer

        model_local_clustering = SemanticLocalClusterizer(
            init_config=self.config.local_clusterizer.init_config,
            gen_config=self.config.local_clusterizer.gen_config,
            session_dir=self.actual_session_dir,
        )
        new_path = model_local_clustering.run(path)
        return new_path

    @check_path_is
    def _call_local_planner(self, path: str | os.PathLike) -> str | os.PathLike:
        """Генерирует микротемы для каждого локального кластера.

        Args:
            path (str | os.PathLike): Путь к локальным хронологическим кластерам.

        Returns:
            str | os.PathLike: Путь к Markdown-артефакту со всеми микротемами лекции.

        Raises:
            Exception: Пробрасывает ошибки генерации или сохранения локального планировщика.
        """
        from src.agents.agent_planner import AgentLocalPlanner

        with AgentLocalPlanner(
            session_dir=self.actual_session_dir,
            init_config=self.config.local_planner.init_config,
            gen_config=self.config.local_planner.gen_config,
            app_config=self.config.local_planner.app_config,
            lecture_theme=self.pipeline_config.lecture_theme,
        ) as planner:
            new_path = planner.run(path)
            return new_path

    @check_path_is
    def _call_global_planner(
        self,
        path: str | os.PathLike,
    ) -> str | os.PathLike:
        """Сворачивает микротемы в глобальный план глав.

        Args:
            path (str | os.PathLike): Путь к артефакту микротем локального планировщика.

        Returns:
            str | os.PathLike: Путь к JSON глобального плана с заголовками
            и описаниями глав.

        Raises:
            Exception: Пробрасывает ошибки генерации или сохранения глобального планировщика.
        """
        from src.agents.agent_planner import AgentGlobalPlanner

        with AgentGlobalPlanner(
            session_dir=self.actual_session_dir,
            init_config=self.config.global_planner.init_config,
            gen_config=self.config.global_planner.gen_config,
            app_config=self.config.global_planner.app_config,
            lecture_theme=self.pipeline_config.lecture_theme,
        ) as planner:
            new_path = planner.run(path)
            return new_path

    @check_path_is
    def _call_planner(
        self,
        path: str | os.PathLike,
    ) -> str | os.PathLike:
        """Запускает локальное, затем глобальное планирование.

        Args:
            path (str | os.PathLike): Путь к тексту локальных кластеров.

        Returns:
            str | os.PathLike: Путь к артефакту глобального плана.

        Raises:
            Exception: Пробрасывает ошибки этапов планирования.
        """
        local_clusters_path = self._call_local_planner(path)
        new_path = self._call_global_planner(local_clusters_path)

        return new_path

    @check_path_is
    def _call_global_clustering(
        self,
        global_plan_path: str | os.PathLike,
        local_clusters_path: str | os.PathLike,
    ) -> str | os.PathLike:
        """Назначает локальные кластеры главам из глобального плана.

        Args:
            global_plan_path (str | os.PathLike): Путь к JSON глобального плана.
            local_clusters_path (str | os.PathLike): Путь к локальным кластерам.

        Returns:
            str | os.PathLike: Путь к глобальным кластерам для синтеза.

        Raises:
            Exception: Пробрасывает ошибки глобальной кластеризации.
        """
        from src.core.clustering import SemanticGlobalClusterizer

        model_global_clustering = SemanticGlobalClusterizer(
            init_config=self.config.global_clusterizer.init_config,
            session_dir=self.actual_session_dir,
        )
        new_path = model_global_clustering.run(global_plan_path, local_clusters_path)
        return new_path

    @check_path_is
    def _call_clustering(
        self,
        path: str | os.PathLike,
    ) -> str | os.PathLike:
        """Запускает кластеризацию транскрипта и распределение по главам.

        Args:
            path (str | os.PathLike): Путь к сырому артефакту транскрипта.

        Returns:
            str | os.PathLike: Путь к глобальным кластерам, разбитым по глобальным темам.

        Raises:
            Exception: Пробрасывает ошибки этапов кластеризации или планирования.
        """
        local_clusters_path = self._call_local_clustering(path)
        plan_path = self._call_planner(local_clusters_path)
        new_path = self._call_global_clustering(plan_path, local_clusters_path)

        return new_path

    @check_path_is
    def _call_synthesizer(
        self,
        path: str | os.PathLike,
    ) -> str | os.PathLike:
        """Запускает финальный синтез конспекта по глобальным кластерам.

        Args:
            path (str | os.PathLike): Путь к глобальным кластерам, сгруппированным по темам.

        Returns:
            str | os.PathLike: Путь к JSON синтезированного конспекта.

        Raises:
            Exception: Пробрасывает ошибки синтезатора или extractor.
        """
        from src.agents.agent_synthesizer import AgentSynthesizerLlama

        with AgentSynthesizerLlama(
            session_dir=self.actual_session_dir,
            extractor_gen_config=self.config.extractor.gen_config,
            extractor_app_config=self.config.extractor.app_config,
            init_config=self.config.synthesizer.init_config,
            gen_config=self.config.synthesizer.gen_config,
            app_config=self.config.synthesizer.app_config,
            lecture_theme=self.pipeline_config.lecture_theme,
        ) as synthesizer:
            new_path = synthesizer.run(path)

        return new_path

    @check_path_is
    def _call_graph_planner(
        self,
        path: str | os.PathLike,
    ) -> str | os.PathLike:
        """Вставляет запросы на графики в Markdown-конспект.

        Args:
            path (str | os.PathLike): Путь к артефакту Markdown-конспекта.

        Returns:
            str | os.PathLike: Путь к Markdown с плейсхолдерами графиков.

        Raises:
            Exception: Пробрасывает ошибки graph planner.
        """
        from src.agents.agent_graph_planner import AgentGraphPlanner

        with AgentGraphPlanner(
            session_dir=self.actual_session_dir,
            init_config=self.config.graph_planner.init_config,
            gen_config=self.config.graph_planner.gen_config,
            app_config=self.config.graph_planner.app_config,
            lecture_theme=self.pipeline_config.lecture_theme,
        ) as graph_planner:
            new_path = graph_planner.run(path)
            return new_path

    # Добавить в utils или создать новую папку хз
    def convert_json_to_md(self, path: str | os.PathLike) -> Path:
        """Преобразует синтезированный JSON-конспект в Markdown.

        Args:
            path (str | os.PathLike): Путь к JSON синтезированного конспекта.

        Returns:
            Path: Путь к артефакту Markdown-конспекта для планирования графиков.

        Raises:
            OSError: Если входной JSON невозможно прочитать или Markdown невозможно сохранить.
            json.JSONDecodeError: Если JSON конспекта невалиден.
        """
        with open(path, "r", encoding="utf-8") as file:
            conspect = json.load(file)

        md_lines = [
            "**Этот конспект сгенерирован с помощью AI.**",
            "**Система может допускать ошибки в формулах, вычислениях и специфической терминологии.**",
            "**Пожалуйста, относитесь с понимаем и проверяйте конспект!**\n",
        ]

        for topic, body in conspect.items():
            md_lines.append(f"# {topic}\n")
            if isinstance(body, list):
                text_body = "\n\n".join(str(item) for item in body)
            else:
                text_body = str(body)
            md_lines.append(f"{text_body}\n")

        final_conspect = "\n".join(md_lines)

        out_filepath = self._safe_result_out_line(
            output=final_conspect,
            stage="07_conspect_md",
            file_name="conspect.md",
            session_dir=self.actual_session_dir,
            extension="md",
        )

        return out_filepath

    @check_path_is
    def _call_grapher(self, path: str | os.PathLike) -> str | os.PathLike:
        """Генерирует артефакты изображений графиков для плейсхолдеров конспекта.

        Args:
            path (str | os.PathLike): Путь к Markdown с плейсхолдерами графиков.

        Returns:
            str | os.PathLike: Путь к JSON результата генерации графиков.

        Raises:
            Exception: Пробрасывает ошибки генерации графиков.
        """
        from src.agents.agent_grapher import AgentGrapher

        with AgentGrapher(
            session_dir=self.actual_session_dir,
            init_config=self.config.grapher.init_config,
            gen_config=self.config.grapher.gen_config,
            app_config=self.config.grapher.app_config,
            lecture_theme=self.pipeline_config.lecture_theme,
            getting_graphs_from_conspect_func=LongConspectWriterPipeline.getting_graphs_from_conspect,
        ) as grapher:
            new_path = grapher.run(path)
            return new_path

    def getting_graphs_from_conspect(
        self,
        conspect: str,
        tag_open: str = "[",
        tag_close: str = "]",
        tag_meat: str = "[GRAPH_TYPE:",
    ) -> list[tuple[int, int, str]]:
        """Ищет плейсхолдеры графиков в конспекте с поддержкой вложенных скобок.

        Args:
            conspect (str): Текст Markdown-конспекта, созданный graph planner.
            tag_open (str): Opening bracket token for graph placeholder scanning.
            tag_close (str): Closing bracket token for graph placeholder scanning.
            tag_meat (str): Prefix that marks a graph placeholder.

        Returns:
            list[tuple[int, int, str]]: Placeholder start index, end index, and
            surrounding context text.

        Raises:
            Exception: Намеренно не выбрасывает исключения.
        """
        graphs = []
        char_open_count = 0
        idx_start = 0

        for i, char in enumerate(conspect):
            if char == tag_open:
                if char_open_count == 0:
                    if conspect[i : i + 12] == tag_meat:
                        char_open_count = 1
                        idx_start = i
                else:
                    char_open_count += 1
            elif char == tag_close:
                if char_open_count > 0:
                    char_open_count -= 1
                    if char_open_count == 0:
                        left_bound = max(0, idx_start - 200)
                        right_bound = min(len(conspect), i + 201)
                        graphs.append((idx_start, i, conspect[left_bound:right_bound]))
        return graphs

    # Добавить в utils или создать новую папку хз
    def add_graph_in_conspect(
        self, graphs_path: str | os.PathLike, conspect_md_path: str
    ) -> str | os.PathLike:
        """Заменяет плейсхолдеры графиков в Markdown на теги сгенерированных изображений.

        Args:
            graphs_path (str | os.PathLike): Путь к JSON результата генерации графиков
                из ``AgentGrapher``.
            conspect_md_path (str): Путь к Markdown-конспекту, содержащему
                плейсхолдеры графиков.

        Returns:
            str | os.PathLike: Путь к финальному Markdown со ссылками на графики.

        Raises:
            OSError: Если JSON графиков, изображения или Markdown невозможно прочитать/записать.
            json.JSONDecodeError: Если JSON результата графиков невалиден.
            KeyError: Если в записях результата графиков нет ожидаемых полей.
        """
        graphs_file_path = Path(graphs_path)
        with open(graphs_file_path, "r", encoding="utf-8") as file:
            graphs = json.load(file)

        with open(conspect_md_path, "r", encoding="utf-8") as file:
            conspect = file.read()

        stage_name = "10_conspect_with_graph_md"
        final_md_dir = self.actual_session_dir / stage_name

        # Создаем папку для картинок рядом с финальным конспектом в новой сессии
        final_assets_dir = final_md_dir / "assets"
        final_assets_dir.mkdir(parents=True, exist_ok=True)

        # Вычисляем директорию, в которой лежит переданный JSON (старая сессия)
        graphs_base_dir = graphs_file_path.parent

        for place_holder, value in graphs.items():
            if value["status"] == "success":
                # Ищем картинку там же, где лежит JSON, а не в новой пустой сессии
                absolute_image_path = graphs_base_dir / value["path"]
                image_name = value["name_graph"]
                formated_image_name = (
                    image_name.split("___")[1].replace("_", " ").replace(".png", "")
                )

                if absolute_image_path.exists():
                    destination_path = final_assets_dir / absolute_image_path.name
                    shutil.copy2(absolute_image_path, destination_path)

                    markdown_valid_path = f"assets/{absolute_image_path.name}"
                    replacement = f"<div align='center'><img src='{markdown_valid_path}' width='700'><br><p>{formated_image_name}</p></div>"
                else:
                    logger.error(f"Файл не найден при сборке: {absolute_image_path}")
                    replacement = (
                        f"*Ошибка сборки: файл {absolute_image_path.name} утерян*"
                    )
            else:
                replacement = f"*Ошибка генерации визуализации для: {place_holder}*"

            conspect = conspect.replace(place_holder, replacement)

        out_filepath = self._safe_result_out_line(
            output=conspect,
            stage=stage_name,
            file_name="final_conspect.md",
            session_dir=self.actual_session_dir,
            extension="md",
        )

        return out_filepath

    def run(self, audio_file_path: str | os.PathLike) -> str | None:
        """Запускает полный последовательный пайплайн от аудио к конспекту.

        Args:
            audio_file_path (str | os.PathLike): Путь к аудио или видео лекции,
                с которого начинается пайплайн.

        Returns:
            str | None: Путь к финальному Markdown-конспекту с графиками или ``None``,
            если проверяемый этап неожиданно не вернул путь.

        Raises:
            Exception: Пробрасывает ошибки любого строгого этапа пайплайна.
        """
        transcript_path = self._call_stt(audio_file_path)

        clustering_path = self._call_clustering(transcript_path)

        conspect_json = self._call_synthesizer(clustering_path)

        conspect_md_path = self.convert_json_to_md(conspect_json)

        conspect_md_path = self._call_graph_planner(conspect_md_path)

        graphs_path = self._call_grapher(path=conspect_md_path)

        conspect_with_graph = self.add_graph_in_conspect(
            graphs_path=graphs_path, conspect_md_path=conspect_md_path
        )

        return conspect_with_graph
