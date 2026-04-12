from src.core.transcribing import FasterWhisper
from src.agents.agent_drafter import AgentDrafter
from src.agents.agent_synthesizer import AgentSynthesizer
from src.agents.agent_planner import AgentLocalPlanner, AgentGlobalPlanner
from src.core.clustering import SemanticLocalClusterizer, SemanticGlobalClusterizer
from loguru import logger
import multiprocessing
from os import PathLike
from src.core.compression import SmartCompressor
from src.agents.base_agent import Trackable


# Раздутый init потом поменять просто на датакласс
class ConspectiusPipeline(Trackable):
    def __init__(
        self,
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

    def _run_stt_process(
        self, path: str | PathLike, result_queue: multiprocessing.Queue
    ) -> None:
        try:
            faster_whisper = FasterWhisper(
                init_config=self.stt_init_config,
                gen_config=self.stt_gen_config,
                app_config=self.stt_app_config,
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

    def _call_local_clustering(self, path: str | PathLike) -> str | PathLike:
        """
        Разбивает сплошной текст на смысловые абзацы.
        Вход (path): Путь к сырой транскрипции (результат STT).
        Выход: Путь к файлу, где текст разделен сепараторами (локальные кластеры).
        """
        # Пока захардкодил. Потом добавить в конфиг
        model_name = "cointegrated/rubert-tiny2"
        model_local_clustering = SemanticLocalClusterizer(model_name)
        new_path = model_local_clustering.run(path)
        return new_path

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
        ) as planner:
            new_path = planner.run(path)
            return new_path

    def _call_global_planner(self, path: str | PathLike) -> str | PathLike:
        """
        Собирает микро-темы в финальное оглавление (LLM Reduce).
        Вход (path): Путь к списку микро-тем (результат LocalPlanner).
        Выход: Путь к JSON-файлу со структурой глав (chapter_title, description).
        """
        with AgentGlobalPlanner(
            init_config=self.global_planner_init_config,
            gen_config=self.global_planner_gen_config,
            app_config=self.global_planner_app_config,
        ) as planner:
            new_path = planner.run(path)
            return new_path

    def _call_planner(self, path: str | PathLike) -> str | PathLike:
        """
        Оркестратор планирования (Local -> Global).
        Вход (path): Путь к локальным кластерам (сырые абзацы).
        Выход: Путь к глобальному плану (JSON-оглавление).
        """
        local_clusters_path = self._call_local_planner(path)
        new_path = self._call_global_planner(local_clusters_path)

        return new_path

    def _call_global_clustering(
        self, global_plan_path: str | PathLike, local_clusters_path: str | PathLike
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
        model_global_clustering = SemanticGlobalClusterizer(model_name)
        new_path = model_global_clustering.run(global_plan_path, local_clusters_path)

        return new_path

    def _call_clustering(self, path: str | PathLike) -> str | PathLike:
        """
        Главный оркестратор всей логики кластеризации текста.
        Вход (path): Путь к сырой транскрипции (результат STT).
        Выход: Путь к глобальным кластерам разбитым по глобальным темам.
        """
        local_clusters_path = self._call_local_clustering(path)
        plan_path = self._call_planner(local_clusters_path)
        new_path = self._call_global_clustering(plan_path, local_clusters_path)

        return new_path

    def _call_drafter(self, path: str | PathLike) -> str | PathLike | None:
        """
        Запускает AgentDrafter для извлечения фактов, определений и теорем.
        Вход (path): Путь к тематическому кластеру (результат глобальной кластеризации).
        Выход: Путь к файлу с мини-конспектами и плейсхолдерами {{formula}}, {{figure}}.
                None если Drafter не нашёл академического контента.
        """
        with AgentDrafter(
            init_config=self.drafter_init_config,
            gen_config=self.drafter_gen_config,
            app_config=self.drafter_app_config,
        ) as drafter:
            new_path = drafter.run(path)
        if not new_path:
            return None
        return new_path

    # Идея передать метод — архитектурно правильная.
    # Ты интуитивно пришел к паттерну Dependency Injection (Внедрение зависимостей) через передачу callback-функции.
    # Для твоего разросшегося пайплайна это единственное адекватное решение.

    # Вот почему это хороший подход:

    #     Слабая связность (Decoupling): SmartCompressor — это маршрутизатор. Он не должен знать, как инициализировать LLM, какие веса грузить и какие конфиги нужны Экстрактору.
    #     Его единственная ответственность — подсчет токенов и принятие решения.

    #     Изолированное тестирование: Чтобы протестировать SmartCompressor, тебе больше не нужны GPU и загрузка моделей.
    #     Ты передаешь в метод обычную mock-функцию (заглушку), которая возвращает строку "сжатый текст", и проверяешь логику if/else.
    #     AgentDrafter при этом тестируется отдельно.

    #     Чистота конструктора: Из SmartCompressor полностью уходят drafter_init_config, drafter_gen_config и drafter_app_config.
    def _call_compression(self, path: str | PathLike) -> str | PathLike:
        """
        Запускает SmartCompressor — промежуточный этап между кластеризацией и синтезом.
        Вход (path): Путь к глобальным кластерам (результат _call_clustering).
        Выход: Путь к сжатому/обработанному файлу готовому для Synthesizer.
        """
        compressor = SmartCompressor(
            synthesizer_init_config=self.synthesizer_init_config,
            synthesizer_gen_config=self.synthesizer_gen_config,
        )
        new_path = compressor.process(path)
        return new_path

    def _call_synthesizer(self, path: str | PathLike) -> str | PathLike:
        """
        Запускает AgentSynthesizer (LongWriter) для синтеза финального конспекта.
        Вход (path): Путь к сжатым мини-конспектам (результат _call_compression).
        Выход: Путь к итоговому файлу конспекта в формате Markdown.
        """
        with AgentSynthesizer(
            init_config=self.synthesizer_init_config,
            gen_config=self.synthesizer_gen_config,
            app_config=self.synthesizer_app_config,
        ) as synthesizer:
            new_path = synthesizer.run(path)
            return new_path

    def run(self, audio_file_path: str | PathLike) -> str | None:
        """
        Главный оркестратор полного пайплайна: аудио → конспект.
        Этапы: STT → локальная кластеризация → планирование →
            глобальная кластеризация → компрессия → синтез.
        Вход (audio_file_path): Путь к аудиофайлу лекции.
        Выход: Путь к итоговому конспекту. None если пайплайн не вернул результат.
        """
        transcript_path = self._call_stt(audio_file_path)
        logger.info(f"Получен путь для транскрибации: {transcript_path}")

        clustering_path = self._call_clustering(transcript_path)

        # Для теста пайплайна убрал пока что
        # compression_path = self._call_compression(clustering_path)

        conspect_path = self._call_synthesizer(clustering_path)

        # if not chunk_conspects_path:
        #     logger.warning("Drafter не вернул ни одного валидного мини-конспекта.")
        #     return None

        # logger.info(f"Получен путь до мини-конспектов: {chunk_conspects_path}")
        # final_conspect_path = self._call_synthesizer(chunk_conspects_path)

        # return final_conspect_path
        return conspect_path
