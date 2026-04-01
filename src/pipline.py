from src.ai_configs import (
    STTModelConfig,
    LLMModelConfig,
    decorator_v_ram_cleaner,
    InitConfig,
    GenConfig,
)
from src.core.transcribing import FasterWhisper
from src.core.agent_drafter import AgentDrafter
from src.core.agent_synthesizer import AgentSynthesizer
from loguru import logger
import multiprocessing


def _run_stt_process(audio_file_path: str, result_queue: multiprocessing.Queue):
    try:
        stt_model_config = STTModelConfig(model_size="large-v3-turbo")
        faster_whisper = FasterWhisper(stt_model_config)
        transcript_path = faster_whisper.run(audio_file_path=audio_file_path)
        result_queue.put({"status": "success", "path": transcript_path})
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})


class ConspectiusPipline:

    @decorator_v_ram_cleaner
    def _call_sst(self, audio_file_path: str) -> str:
        logger.info("Запуск STT агента в изолированном процессе...")
        # Создаем канал связи (Очередь) между основным и фоновым процессом
        result_queue = multiprocessing.Queue()
        # Настраиваем фоновый процесс
        process = multiprocessing.Process(
            target=_run_stt_process, args=(audio_file_path, result_queue)
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

    @decorator_v_ram_cleaner
    def _call_drafter(self, path_transcrib: str):
        drafter_init_config = InitConfig(model="Qwen/Qwen2.5-7B-Instruct")
        drafter_gen_config = GenConfig()

        with AgentDrafter(
            init_config=drafter_init_config, gen_config=drafter_gen_config
        ) as drafter:
            chunk_conspects_path = drafter.run(path_transcrib)
            return chunk_conspects_path

    @decorator_v_ram_cleaner
    def _call_synthesizer(self, chunk_conspects_path: str):
        synthesizer_init_config = InitConfig(model="THUDM/LongWriter-llama3.1-8b")
        synthesizer_gen_config = GenConfig()

        with AgentSynthesizer(
            init_config=synthesizer_init_config, gen_config=synthesizer_gen_config
        ) as synthesizer:
            final_conspect_path = synthesizer.run(chunk_conspects_path)
            return final_conspect_path

    def pipline(self, audio_file_path: str | None = None):
        # # Это нужно для тестирования _call_drafter и _call_synthesizer
        if audio_file_path is None:
            # transcript_path = r"data\example-transcrib\large-v3-turbo-cuda-float16-Защита инф-1774820035.txt"
            transcript_path = r"data\example-transcrib\large-v3-turbo-cuda-float16-Защита инф-1775083819.txt"
        else:
            transcript_path = self._call_sst(audio_file_path)
            logger.success(f"Получен путь для транскрибации: {transcript_path}")

        chunk_conspects_path = self._call_drafter(transcript_path)
        logger.success(f"Получен путь до мини-конспектов: {transcript_path}")

        final_conspect_path = self._call_synthesizer(chunk_conspects_path)
        logger.success(f"Получен путь до финального конспекта: {final_conspect_path}")

        return final_conspect_path
