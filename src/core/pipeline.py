"""Оркестрация последовательного пайплайна LongConspectWriter.

Модуль связывает STT, кластеризацию, планирование, синтез, планирование
графиков, рендер графиков и финальную сборку Markdown в релизный пайплайн.
Каждый метод сохраняет существующий контракт передачи путей между этапами.
"""

import os
from src.core.base import BasePipeline
from src.core.utils import check_path_is
from loguru import logger
from src.configs.configs import PipelineSessionConfig
import json
import shutil
from pathlib import Path
import markdown


class LongConspectWriterPipeline(BasePipeline):
    """Координирует все этапы LongConspectWriter в строгом порядке."""

    def __init__(self, session_config: PipelineSessionConfig) -> None:
        """Инициализирует запуск пайплайна и создает директорию сессии.

        Args:
            session_config (PipelineSessionConfig): Полностью загруженная конфигурация
                для каждого этапа LongConspectWriter.

        Returns:
            None: Пайплайн сохраняет конфиг и состояние активной сессии.
        """
        self.config = session_config
        self.pipeline_config = self.config.pipeline
        self.__post_init__()

    @check_path_is
    def _call_stt(self, path: str | os.PathLike) -> str:
        """Запускает этап транскрибации FasterWhisper в изолированном subprocess.

        Args:
            path (str | os.PathLike): Путь к исходному аудио или видео лекции.

        Returns:
            str: Путь к сырому артефакту транскрипта, созданному STT.
        """
        from src.core.stt import FasterWhisper

        return FasterWhisper(
            session_dir=self.actual_session_dir,
            init_config=self.config.stt.init_config,
            gen_config=self.config.stt.gen_config,
            app_config=self.config.stt.app_config,
            lecture_theme=self.pipeline_config.lecture_theme,
        ).run(audio_file_path=path)

    @check_path_is
    def _call_local_clustering(self, path: str | os.PathLike) -> str | os.PathLike:
        """Запускает локальную хронологическую кластеризацию по артефакту транскрипта.

        Args:
            path (str | os.PathLike): Путь к артефакту транскрипта.

        Returns:
            str | os.PathLike: Путь к локальным кластерам для этапов планирования.
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

    @check_path_is
    def convert_json_to_md(self, path: str | os.PathLike) -> Path:
        """Преобразует синтезированный JSON-конспект в Markdown.

        Args:
            path (str | os.PathLike): Путь к JSON синтезированного конспекта.

        Returns:
            Path: Путь к артефакту Markdown-конспекта для планирования графиков.
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
            tag_open (str): Токен открывающей скобки при сканировании плейсхолдеров.
            tag_close (str): Токен закрывающей скобки при сканировании плейсхолдеров.
            tag_meat (str): Префикс, маркирующий плейсхолдер графика.

        Returns:
            list[tuple[int, int, str]]: Начальный индекс, конечный индекс плейсхолдера
            и окружающий контекстный текст.
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

    @check_path_is
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

    @check_path_is
    def convert_md_to_pdf(self, path: str | os.PathLike) -> Path:
        """Конвертирует финальный Markdown-конспект в PDF с поддержкой LaTeX и графиков.

        Args:
            path (str | os.PathLike): Путь к финальному Markdown-конспекту.

        Returns:
            Path: Путь к сгенерированному PDF-артефакту.
        """
        import playwright.sync_api

        md_path = Path(path).resolve()
        md_text = md_path.read_text(encoding="utf-8")

        body = markdown.markdown(
            md_text,
            extensions=["extra", "codehilite", "tables", "fenced_code"],
        )

        html = f"""<!DOCTYPE html>
    <html lang="ru">
    <head>
    <meta charset="utf-8">
    <script>
    window.MathJax = {{
    tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
    }},
    startup: {{
        pageReady: () => MathJax.startup.defaultPageReady().then(() => {{
        window.mathJaxDone = true;
        }})
    }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
    body {{
        font-family: 'Segoe UI', 'DejaVu Sans', sans-serif;
        max-width: 800px;
        margin: 2em auto;
        line-height: 1.6;
        padding: 0 1em;
    }}
    pre {{ background: #f4f4f4; padding: 1em; border-radius: 4px; overflow-x: auto; }}
    code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
    pre code {{ background: transparent; padding: 0; }}
    img {{ max-width: 100%; height: auto; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
    h1, h2, h3 {{ page-break-after: avoid; }}
    pre, table, img {{ page-break-inside: avoid; }}
    </style>
    </head>
    <body>
    {body}
    </body>
    </html>"""

        stage_name = "11_conspect_pdf"
        pdf_dir = self.actual_session_dir / stage_name
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "final_conspect.pdf"

        base_url = md_path.parent.as_uri() + "/"

        with playwright.sync_api.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(base_url, wait_until="domcontentloaded")
            page.set_content(html, wait_until="networkidle")
            page.evaluate("""
                            () => new Promise((resolve, reject) => {
                                let attempts = 0;
                                const maxAttempts = 100; // 10 секунд (100 * 100ms)
                                
                                const check = () => {
                                    if (window.mathJaxDone) {
                                        resolve();
                                    } else if (attempts >= maxAttempts) {
                                        reject(new Error("Таймаут: MathJax не отрендерил формулы за 10 секунд. Проверьте CDN или интернет."));
                                    } else {
                                        attempts++;
                                        setTimeout(check, 100);
                                    }
                                };
                                check();
                            })
                        """)
            page.pdf(
                path=str(pdf_path),
                format="A4",
                margin={
                    "top": "20mm",
                    "bottom": "20mm",
                    "left": "15mm",
                    "right": "15mm",
                },
                print_background=True,
            )
            browser.close()

        logger.success(f"PDF сгенерирован: {pdf_path}")
        return pdf_path

    @check_path_is
    def run(self, audio_file_path: str | os.PathLike) -> str | None:
        """Запускает полный последовательный пайплайн от аудио к конспекту.

        Args:
            audio_file_path (str | os.PathLike): Путь к аудио или видео лекции,
                с которого начинается пайплайн.

        Returns:
            str | None: Путь к финальному Markdown-конспекту с графиками или ``None``,
            если проверяемый этап неожиданно не вернул путь.
        """
        self._run_meta["audio_file"] = str(audio_file_path)
        self._run_meta["lecture_theme"] = self.pipeline_config.lecture_theme
        self._write_root_meta()

        try:
            transcript_path = self._timed_call(
                "01_stt", self._call_stt, audio_file_path
            )
            clustering_path = self._timed_call(
                "05_global_clusters", self._call_clustering, transcript_path
            )
            conspect_json = self._timed_call(
                "06_synthesizer", self._call_synthesizer, clustering_path
            )
            conspect_md_path = self._timed_call(
                "07_conspect_md", self.convert_json_to_md, conspect_json
            )
            conspect_md_path = self._timed_call(
                "08_graph_planner", self._call_graph_planner, conspect_md_path
            )
            graphs_path = self._timed_call(
                "09_grapher", self._call_grapher, path=conspect_md_path
            )
            conspect_with_graph = self._timed_call(
                "10_conspect_with_graph_md",
                self.add_graph_in_conspect,
                graphs_path=graphs_path,
                conspect_md_path=conspect_md_path,
            )
            pdf_path = self._timed_call(
                "11_conspect_pdf", self.convert_md_to_pdf, conspect_with_graph
            )
            self._finish_run("success")
            return pdf_path
        except BaseException:
            self._finish_run("failed")
            raise
