from src.agents.transcribing import FasterWhisper
from src.agents.agent_drafter import AgentDrafter
from src.agents.agent_synthesizer import AgentSynthesizer
from loguru import logger
import multiprocessing
from os import PathLike


class ConspectiusPipeline:
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

    def _run_stt_process(
        self, audio_file_path: str | PathLike, result_queue: multiprocessing.Queue
    ) -> None:
        try:
            faster_whisper = FasterWhisper(
                init_config=self.stt_init_config,
                gen_config=self.stt_gen_config,
                app_config=self.stt_app_config,
            )
            transcript_path = faster_whisper.run(audio_file_path=audio_file_path)
            result_queue.put({"status": "success", "path": transcript_path})
        except Exception as e:
            result_queue.put({"status": "error", "error": str(e)})

    def _call_stt(self, audio_file_path: str | PathLike) -> str:

        logger.info("Запуск STT агента в изолированном процессе...")
        # Создаем канал связи (Очередь) между основным и фоновым процессом
        result_queue = multiprocessing.Queue()
        # Настраиваем фоновый процесс
        process = multiprocessing.Process(
            target=self._run_stt_process, args=(audio_file_path, result_queue)
        )
        # Запускаем его
        process.start()
        # Основная программа ждет, пока процесс STT не завершится
        process.join()
        # Проверяем, что нам вернул процесс
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

    def _call_drafter(self, path_transcrib: str | PathLike) -> str | PathLike | None:
        with AgentDrafter(
            init_config=self.drafter_init_config,
            gen_config=self.drafter_gen_config,
            app_config=self.drafter_app_config,
        ) as drafter:
            chunk_conspects_path = drafter.run(path_transcrib)
        if not chunk_conspects_path:
            return None
        return chunk_conspects_path

    def _call_synthesizer(self, chunk_conspects_path: str | PathLike) -> str | PathLike:
        with AgentSynthesizer(
            init_config=self.synthesizer_init_config,
            gen_config=self.synthesizer_gen_config,
            app_config=self.synthesizer_app_config,
        ) as synthesizer:
            final_conspect_path = synthesizer.run(chunk_conspects_path)
            return final_conspect_path

    def pipeline(self, audio_file_path: str | PathLike) -> str | None:
        transcript_path = self._call_stt(audio_file_path)
        logger.info(f"Получен путь для транскрибации: {transcript_path}")

        chunk_conspects_path = self._call_drafter(transcript_path)

        if not chunk_conspects_path:
            logger.warning("Drafter не вернул ни одного валидного мини-конспекта.")
            return None

        logger.info(f"Получен путь до мини-конспектов: {chunk_conspects_path}")
        final_conspect_path = self._call_synthesizer(chunk_conspects_path)

        return final_conspect_path
