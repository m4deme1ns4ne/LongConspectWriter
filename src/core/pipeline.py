from src.core.stt import FasterWhisper
from src.agents.agent_planner import AgentLocalPlanner, AgentGlobalPlanner
from src.core.clustering import SemanticLocalClusterizer, SemanticGlobalClusterizer
from os import PathLike
from src.core.base import BasePipeline
from src.core.utils import check_path_is
from loguru import logger
import multiprocessing
from src.agents.agent_synthesizer import AgentSynthesizerLlama
import json


# Раздутый init потом поменять просто на датакласс
class LongConspectWriterPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config,
        stt_init_config,
        stt_gen_config,
        stt_app_config,
        drafter_init_config,
        drafter_gen_config,
        drafter_app_config,
        synthesizer_init_config,
        synthesizer_gen_config,
        synthesizer_app_config,
        local_planner_init_config,
        local_planner_gen_config,
        local_planner_app_config,
        global_planner_init_config,
        global_planner_gen_config,
        global_planner_app_config,
    ):
        self.pipeline_config = pipeline_config
        self.__post_init__()

        self.stt_init_config = stt_init_config
        self.stt_gen_config = stt_gen_config
        self.stt_app_config = stt_app_config

        self.drafter_init_config = drafter_init_config
        self.drafter_gen_config = drafter_gen_config
        self.drafter_app_config = drafter_app_config

        self.synthesizer_init_config = synthesizer_init_config
        self.synthesizer_gen_config = synthesizer_gen_config
        self.synthesizer_app_config = synthesizer_app_config

        self.local_planner_init_config = local_planner_init_config
        self.local_planner_gen_config = local_planner_gen_config
        self.local_planner_app_config = local_planner_app_config

        self.global_planner_init_config = global_planner_init_config
        self.global_planner_gen_config = global_planner_gen_config
        self.global_planner_app_config = global_planner_app_config

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
            faster_whisper = FasterWhisper(
                init_config=self.stt_init_config,
                gen_config=self.stt_gen_config,
                app_config=self.stt_app_config,
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

    # @check_path_is
    # def _call_drafter(self, path: str | PathLike) -> str | PathLike | None:
    #     """
    #     Запускает AgentDrafter для извлечения фактов, определений и теорем.
    #     Вход (path): Путь к тематическому кластеру (результат глобальной кластеризации).
    #     Выход: Путь к файлу с мини-конспектами и плейсхолдерами {{formula}}, {{figure}}.
    #             None если Drafter не нашёл академического контента.
    #     """
    #     backend = getattr(self.drafter_init_config, "backend", "llamacpp")

    #     if backend == "llamacpp":
    #         logger.info("Запуск Драфтера через Llama.cpp (Оптимальный режим)")
    #         from src.agents.agent_drafter import AgentDrafterLlama

    #         with AgentDrafterLlama(
    #             init_config=self.drafter_init_config,
    #             gen_config=self.drafter_gen_config,
    #             app_config=self.drafter_app_config,
    #         ) as drafter:
    #             new_path = drafter.run(path)

    #     elif backend == "transformers":
    #         logger.info("Запуск Драфтера через Transformers (Режим совместимости)")
    #         from src.agents.agent_drafter import AgentDrafterTransformers

    #         with AgentDrafterTransformers(
    #             init_config=self.drafter_init_config,
    #             gen_config=self.drafter_gen_config,
    #             app_config=self.drafter_app_config,
    #         ) as drafter:
    #             new_path = drafter.run(path)
    #     else:
    #         raise ValueError(f"Неизвестный бэкенд для Синтезатора: {backend}")

    #     return new_path

    @check_path_is
    def _call_local_clustering(self, path: str | PathLike) -> str | PathLike:
        """
        Разбивает сплошной текст на смысловые абзацы.
        Вход (path): Путь к сырой транскрипции (результат STT).
        Выход: Путь к файлу, где текст разделен сепараторами (локальные кластеры).
        """
        # Пока захардкодил. Потом добавить в конфиг
        model_name = "cointegrated/rubert-tiny2"
        model_local_clustering = SemanticLocalClusterizer(
            model_name, session_dir=self.actual_session_dir
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
        with AgentLocalPlanner(
            init_config=self.local_planner_init_config,
            gen_config=self.local_planner_gen_config,
            app_config=self.local_planner_app_config,
            session_dir=self.actual_session_dir,
        ) as planner:
            new_path = planner.run(path)
            return new_path

    # Тут надо будеь пошаманить, чтобы он выдавал идельное кол во глав, может как то математически расчитывал...
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
        with AgentGlobalPlanner(
            init_config=self.global_planner_init_config,
            gen_config=self.global_planner_gen_config,
            app_config=self.global_planner_app_config,
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
        # Пока захардкодил. Потом добавить в конфиг
        model_name = "intfloat/multilingual-e5-small"
        model_global_clustering = SemanticGlobalClusterizer(
            model_name, session_dir=self.actual_session_dir
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
        Запускает AgentSynthesizer (LongWriter) для синтеза финального конспекта.
        """

        logger.info("Запуск Синтезатора через Llama.cpp (Оптимальный режим)")
        with AgentSynthesizerLlama(
            init_config=self.synthesizer_init_config,
            gen_config=self.synthesizer_gen_config,
            app_config=self.synthesizer_app_config,
            session_dir=self.actual_session_dir,
        ) as synthesizer:
            new_path = synthesizer.run(path)

        return new_path

    def convert_json_to_md(self, path):
        with open(path, "r", encoding="utf-8") as file:
            conspect = json.load(file)

            final_conspect = """**Этот конспект сгенерирован с помощью AI.**\n**Система может допускать ошибки в формулах, вычислениях и специфической терминологии.**\n**Пожалуйста, относитесь с понимаем и проверяйте конспект!**\n\n"""

            for topik, body in conspect.items():
                final_conspect += f"# {topik}\n\n"
                if isinstance(body, list):
                    text_body = "\n\n".join(str(item) for item in body)
                else:
                    text_body = str(body)
                final_conspect += f"{text_body}\n\n"

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

        # compressed_transcript_path = self._call_drafter(transcript_path)

        clustering_path = self._call_clustering(transcript_path)

        conspect_path = self._call_synthesizer(clustering_path)

        final_conspect_path = self.convert_json_to_md(path=conspect_path)

        return final_conspect_path
