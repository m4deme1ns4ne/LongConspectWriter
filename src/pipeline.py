from src.ai_configs import STTModelConfig, InitConfig, GenConfig
from src.core.transcribing import FasterWhisper
from src.core.agent_drafter import AgentDrafter
from src.core.agent_synthesizer import AgentSynthesizer
from loguru import logger
import multiprocessing
from pathlib import Path
from os import PathLike


class ConspectiusPipeline:
    def _run_stt_process(
        self, audio_file_path: str | PathLike, result_queue: multiprocessing.Queue
    ) -> None:
        try:
            stt_model_config = STTModelConfig(model_size="large-v3-turbo")
            faster_whisper = FasterWhisper(stt_model_config)
            transcript_path = faster_whisper.run(audio_file_path=audio_file_path, language_audio="ru")
            result_queue.put({"status": "success", "path": transcript_path})
        except Exception as e:
            result_queue.put({"status": "error", "error": str(e)})


    def _call_sst(self, audio_file_path: str | PathLike) -> str:

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

    def _call_drafter(self, path_transcrib: str | PathLike) -> str:
        drafter_init_config = InitConfig(
            model="Qwen/Qwen2.5-7B-Instruct",
            agent_name="drafter",
            prompt=Path("src") / ("core") / "prompts.yaml",
        )
        drafter_gen_config = GenConfig(
            max_new_tokens=500,
            repetition_penalty=1.15,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )

        with AgentDrafter(
            init_config=drafter_init_config, gen_config=drafter_gen_config
        ) as drafter:
            chunk_conspects_path = drafter.run(path_transcrib)
            return chunk_conspects_path

    def _call_synthesizer(self, chunk_conspects_path: str | PathLike) -> str:
        synthesizer_init_config = InitConfig(
            model="THUDM/LongWriter-llama3.1-8b",
            agent_name="synthesizer",
            prompt=Path("src") / ("core") / "prompts.yaml",
        )
        synthesizer_gen_config = GenConfig(
            max_new_tokens=4096,
            repetition_penalty=1.05,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
        )

        with AgentSynthesizer(
            init_config=synthesizer_init_config, gen_config=synthesizer_gen_config
        ) as synthesizer:
            final_conspect_path = synthesizer.run(chunk_conspects_path)
            return final_conspect_path


    def pipeline(self, audio_file_path: str | PathLike | None = None) -> str | None:
        # Это нужно для тестирования _call_drafter и _call_synthesizer
        if audio_file_path is None:
            litle_transcript_path = r"data\example-transcrib\large-v3-turbo-cuda-float16-Защита инф-1775083687.txt"
            # full_transcript_path = r"data\example-transcrib\large-v3-turbo-cuda-float16-Защита инф-1775162216.txt"
            transcript_path = litle_transcript_path
        else:
            transcript_path = self._call_sst(audio_file_path)
            logger.success(f"Получен путь для транскрибации: {transcript_path}")

        chunk_conspects_path = self._call_drafter(transcript_path)
        if not chunk_conspects_path:
            return

        chunk_conspects_path = r"data\example-mini-conspect\Qwen_Qwen2.5-7B-Instruct-large-v3-turbo-cuda-float16-Лекция 1. -1775312903.txt-1775315971.txt"

        logger.success(f"Получен путь до мини-конспектов: {chunk_conspects_path} ")
        final_conspect_path = self._call_synthesizer(chunk_conspects_path)

        return final_conspect_path
