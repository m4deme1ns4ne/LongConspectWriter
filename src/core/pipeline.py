from os import PathLike
from src.core.base import BasePipeline
from src.core.utils import check_path_is
from loguru import logger
import multiprocessing
import json
from src.configs.configs import PipelineSessionConfig


class LongConspectWriterPipeline(BasePipeline):
    def __init__(self, session_config: PipelineSessionConfig):
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
        self, path: str | PathLike, result_queue: multiprocessing.Queue
    ) -> None:
        try:
            from src.core.stt import FasterWhisper

            faster_whisper = FasterWhisper(
                init_config=self.config.stt.init_config,
                gen_config=self.config.stt.gen_config,
                app_config=self.config.stt.app_config,
                session_dir=self.actual_session_dir,
            )
            transcript_path = faster_whisper.run(audio_file_path=path)
            result_queue.put({"status": "success", "path": transcript_path})
        except Exception as e:
            result_queue.put({"status": "error", "error": str(e)})

    def _call_stt(self, path: str | PathLike) -> str:
        """
        Запускает модель транскрибации (FasterWhisper).
        Вход (path): Путь к исходному аудио или видео файлу.
        Выход: Путь к текстовому файлу (.txt) с сырой транскрипцией.
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
    def _call_local_clustering(self, path: str | PathLike) -> str | PathLike:
        """
        Разбивает сплошной текст на смысловые абзацы.
        Вход (path): Путь к сырой транскрипции (результат STT).
        Выход: Путь к файлу, где текст разделен сепараторами (локальные кластеры).
        """
        from src.core.clustering import SemanticLocalClusterizer

        model_local_clustering = SemanticLocalClusterizer(
            self.pipeline_config.local_clustering_model,
            session_dir=self.actual_session_dir,
        )
        new_path = model_local_clustering.run(path)
        return new_path

    @check_path_is
    def _call_local_planner(self, path: str | PathLike) -> str | PathLike:
        """
        Генерирует микро-темы для каждого абзаца на основе хронологических кластеров.
        Вход (path): Путь к локальным кластерам.
        Выход: Путь к файлу (.md) с плоским списком всех микро-тем лекции.
        """
        from src.agents.agent_planner import AgentLocalPlanner

        with AgentLocalPlanner(
            init_config=self.config.local_planner.init_config,
            gen_config=self.config.local_planner.gen_config,
            app_config=self.config.local_planner.app_config,
            session_dir=self.actual_session_dir,
        ) as planner:
            new_path = planner.run(path)
            return new_path

    @check_path_is
    def _call_global_planner(
        self,
        path: str | PathLike,
    ) -> str | PathLike:
        """
        Собирает микро-темы в финальное оглавление (LLM Reduce).
        Вход (path): Путь к списку микро-тем (результат LocalPlanner).
        Выход: Путь к JSON-файлу со структурой глав (chapter_title, description).
        """
        from src.agents.agent_planner import AgentGlobalPlanner

        with AgentGlobalPlanner(
            init_config=self.config.global_planner.init_config,
            gen_config=self.config.global_planner.gen_config,
            app_config=self.config.global_planner.app_config,
            session_dir=self.actual_session_dir,
        ) as planner:
            new_path = planner.run(path)
            return new_path

    @check_path_is
    def _call_planner(
        self,
        path: str | PathLike,
    ) -> str | PathLike:
        """
        Оркестратор планирования (Local -> Global).
        Вход (path): Путь к локальным кластерам (сырые абзацы).
        Выход: Путь к глобальному плану (JSON-оглавление).
        """
        local_clusters_path = self._call_local_planner(path)
        new_path = self._call_global_planner(local_clusters_path)

        return new_path

    @check_path_is
    def _call_global_clustering(
        self,
        global_plan_path: str | PathLike,
        local_clusters_path: str | PathLike,
    ) -> str | PathLike:
        """
        Связывает оригинальные абзацы с главами оглавления через косинусное сходство.
        Вход:
          - global_plan_path: Путь к JSON-оглавлению.
          - local_clusters_path: Путь к оригинальным локальным кластерам (абзацам).
        Выход: Путь к глобальным кластерам (Json)
        """
        from src.core.clustering import SemanticGlobalClusterizer

        model_global_clustering = SemanticGlobalClusterizer(
            self.pipeline_config.global_clustering_model,
            session_dir=self.actual_session_dir,
        )
        new_path = model_global_clustering.run(global_plan_path, local_clusters_path)

        return new_path

    @check_path_is
    def _call_clustering(
        self,
        path: str | PathLike,
    ) -> str | PathLike:
        """
        Главный оркестратор всей логики кластеризации текста.
        Вход (path): Путь к сырой транскрипции (результат STT).
        Выход: Путь к глобальным кластерам разбитым по глобальным темам.
        """
        local_clusters_path = self._call_local_clustering(path)
        plan_path = self._call_planner(local_clusters_path)
        new_path = self._call_global_clustering(plan_path, local_clusters_path)

        return new_path

    @check_path_is
    def _call_synthesizer(
        self,
        path: str | PathLike,
    ) -> str | PathLike:
        """
        Запускает AgentSynthesizer для синтеза финального конспекта.
        """

        from src.agents.agent_synthesizer import AgentSynthesizerLlama

        with AgentSynthesizerLlama(
            init_config=self.config.synthesizer.init_config,
            gen_config=self.config.synthesizer.gen_config,
            app_config=self.config.synthesizer.app_config,
            session_dir=self.actual_session_dir,
        ) as synthesizer:
            new_path = synthesizer.run(path)

        return new_path

    def convert_json_to_md(self, path):
        with open(path, "r", encoding="utf-8") as file:
            conspect = json.load(file)

        md_lines = [
            "**Этот конспект сгенерирован с помощью AI.**",
            "**Система может допускать ошибки в формулах, вычислениях и специфической терминологии.**",
            "**Пожалуйста, относитесь с понимаем и проверяйте конспект!**\n",
        ]

        for topik, body in conspect.items():
            md_lines.append(f"# {topik}\n")
            if isinstance(body, list):
                text_body = "\n\n".join(str(item) for item in body)
            else:
                text_body = str(body)
            md_lines.append(f"{text_body}\n")

        final_conspect = "\n".join(md_lines)

        out_filepath = self._safe_result_out_line(
            output_dict=final_conspect,
            stage="07_conspect_md",
            file_name="conspect.md",
            session_dir=self.actual_session_dir,
            extension="md",
        )

        return out_filepath

    def run(self, audio_file_path: str | PathLike) -> str | None:
        """
        Главный оркестратор полного пайплайна: аудио → конспект.
        Этапы: STT → локальная кластеризация → планирование →
            глобальная кластеризация → компрессия → синтез.
        Вход (audio_file_path): Путь к аудиофайлу лекции.
        Выход: Путь к итоговому конспекту. None если пайплайн не вернул результат.
        """
        transcript_path = self._call_stt(audio_file_path)

        clustering_path = self._call_clustering(transcript_path)

        conspect_path = self._call_synthesizer(clustering_path)

        final_conspect_path = self.convert_json_to_md(path=conspect_path)

        return final_conspect_path
