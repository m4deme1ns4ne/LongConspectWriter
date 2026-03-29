import time
import os
from faster_whisper import WhisperModel
from loguru import logger

from src.ai_configs import STTModelConfig, AIModelConfig


class FasterWhisper:
    """Агент для локальной транскрибации аудио с помощью faster-whisper."""

    def __init__(self, config: STTModelConfig):
        self._config = config
        logger.success(f"Инициализация агента STT (Модель: {self._config.model_size}, Устройство: {self._config.device})")
    
    def transcribing(self, audio_file_path: str, output_dir: str = "data/example-transcrib",
                           language_audio: str | None = None) -> None:
        """
        Транскрибирует аудиофайл в текст.

        Args:
            audio_file_path (str): Путь к аудиофайлу для транскрибации.
            output_dir (str): Папка для сохранения результатов.
            language_audio (str | None): Язык аудиофайла, если значение None, то модель сама распознает язык.

        Returns:
            None
        """
        if not os.path.exists(audio_file_path):
            logger.error(f"Файл не найден: {audio_file_path}")
            raise FileNotFoundError(f"Файла {audio_file_path}")
        os.makedirs(output_dir, exist_ok=True)
        logger.success(f"Директория {output_dir} найдена!")

        logger.info("Загрузка модели в память...")
        model = WhisperModel(self._config.model_size, device=self._config.device, compute_type=self._config.compute_type)
        logger.success("Модель успешно загружена!")

        timestamp = int(time.time())
        pure_audio_file = os.path.basename(audio_file_path)
        transcrib_file_name = f"{self._config.model_size}-{self._config.device}-{self._config.compute_type}-{pure_audio_file[:10]}-{timestamp}.txt"
        logger.info(f"Итоговое название файла: {transcrib_file_name}")

        logger.info(f"Начинаем транскрибацию файла: {audio_file_path}...")
        start_time = time.time()

        # Метод transcribe автоматически режет длинное аудио на правильные куски
        segments, info = model.transcribe(
            audio_file_path,
            beam_size=5,          # Баланс между скоростью и качеством (5 - стандарт)
            language=language_audio,        # Жестко задаем язык, чтобы модель не тратила время на определение
            vad_filter=True,      # Включаем детектор голоса (игнорирует тишину)
            vad_parameters=dict(min_silence_duration_ms=500) # Настройка чувствительности тишины
        )

        logger.info(f"Определен язык: {info.language} с вероятностью {info.language_probability:.2f}")

        with open(f"{output_dir}/{transcrib_file_name}", "x", encoding="utf-8") as file:
            for segment in segments:
                text = segment.text
                file.write(f"{text}\n")
                file.flush()
                logger.warning(f"Записан новый сегмент: {text[:10] + '...' if len(text) > 10 else text}")


        end_time = time.time()
        logger.success(f"Транскрибация завершена за {end_time - start_time:.2f} секунд.")
