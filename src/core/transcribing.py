import time
import os
import sys
from os import PathLike
from faster_whisper import WhisperModel
from loguru import logger
from tqdm import tqdm
from src.core.base_agent import BaseSTTAgent
from src.ai_configs import STTInitConfig, STTGenConfig


class FasterWhisper(BaseSTTAgent):
    """Агент для локальной транскрибации аудио с помощью faster-whisper."""

    def __init__(self, init_config: STTInitConfig, gen_config: STTGenConfig) -> None:
        self._init_config = init_config
        self._gen_config = gen_config
        logger.info(
            f"Инициализация агента STT (Модель: {self._init_config.model_size_or_path}, Устройство: {self._init_config.device})"
        )
        logger.info(f"Загрузка {self._init_config.model_size_or_path} в память...")
        self.model = WhisperModel(
            self._init_config.model_size_or_path,
            **self._init_config.model_kwargs(),
        )
        logger.info(f"Модель {self._init_config.model_size_or_path} загружена.")

    def run(
        self,
        audio_file_path: str | PathLike,
        language_audio: str | None = None,
    ) -> str:
        """
        Транскрибирует аудиофайл в текст.

        Args:
            audio_file_path (str): Путь к аудиофайлу для транскрибации.
            output_dir (str): Папка для сохранения результатов.
            language_audio (str | None): Язык аудиофайла, если значение None, то модель сама распознает язык.

        Returns:
            str: Путь к сохраненному файлу транскрипции.
        """
        if not os.path.exists(audio_file_path):
            logger.error(f"Файл не найден: {audio_file_path}")
            raise FileNotFoundError(f"Файла {audio_file_path}")
        os.makedirs(self._init_config.output_dir, exist_ok=True)
        logger.debug(f"Каталог для транскрибации готов: {self._init_config.output_dir}")

        timestamp = int(time.time())
        pure_audio_file_name = os.path.basename(audio_file_path)
        transcrib_file_name = f"{self._init_config.model_size_or_path}-{self._init_config.device}-{self._init_config.compute_type}-{pure_audio_file_name[:10]}-{timestamp}.txt"
        logger.debug(f"Итоговое название файла: {transcrib_file_name}")

        logger.info(f"Начинаем обработку аудиофайла файла: {audio_file_path}...")
        start_time = time.time()

        # Метод transcribe автоматически режет длинное аудио на правильные куски
        transcribe_kwargs = self._gen_config.transcribe_kwargs()
        if language_audio is not None:
            transcribe_kwargs["language"] = language_audio

        segments, info = self.model.transcribe(
            audio=audio_file_path,
            **transcribe_kwargs,
        )

        if info.all_language_probs:
            logger.info(
                f"Определен язык: {info.language} с вероятностью {info.language_probability:.2f}"
            )
            top_3_langs = [
                (lang, round(prob, 2)) for lang, prob in info.all_language_probs[:3]
            ]
            logger.debug(f"Топ-3 альтернативных языков: {top_3_langs}")
        else:
            logger.info(
                f"Язык был задан явно, вероятности альтернатив недоступны: {language_audio}"
            )

        # Оставляем параметры для дебага
        logger.debug(f"Параметры транскрибации: {info.transcription_options}")

        # Считаем сэкономленное время
        duration_mins = info.duration / 60
        logger.info(f"Исходная длительность аудио: {duration_mins:.2f} мин.")

        # faster-whisper отдает duration_after_vad только если включен vad_filter=True
        if info.duration_after_vad is not None:
            vad_mins = info.duration_after_vad / 60
            saved_mins = duration_mins - vad_mins
            logger.info(
                f"Длительность полезного сигнала (без тишины): {vad_mins:.2f} мин."
            )
            logger.info(f"VAD-фильтр вырезал {saved_mins:.2f} мин. тишины.")

        final_path = os.path.join(self._init_config.output_dir, transcrib_file_name)

        with open(final_path, "x", encoding="utf-8") as file:
            with tqdm(
                total=round(info.duration, 2),
                unit=" аудио-сек",
                desc="Транскрибация",
                colour="green",
                file=sys.stdout,
                dynamic_ncols=True,
            ) as pbar:
                for segment in segments:
                    text = segment.text.strip()
                    file.write(f"{text}\n")

                    file.flush()
                    os.fsync(file.fileno())

                    progress = segment.end - pbar.n
                    pbar.update(progress)

        end_time = time.time()
        logger.success(
            f"Транскрибация завершена за {end_time - start_time:.2f} секунд."
        )

        return final_path
