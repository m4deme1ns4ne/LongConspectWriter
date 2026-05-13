"""Этап speech-to-text для пайплайна LongConspectWriter.

Модуль оборачивает транскрибацию FasterWhisper, сохраняет артефакт сырого
транскрипта и возвращает его путь следующему этапу кластеризации.
"""

import os
import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from src.core.base import BaseSTTAgent
from dataclasses import asdict


class FasterWhisper(BaseSTTAgent):
    """Реализация первого этапа пайплайна на FasterWhisper."""

    def __init__(self, session_dir: Path, **kwargs: object) -> None:
        """Инициализирует STT-агента для текущей сессии пайплайна.

        Args:
            session_dir (Path): Директория текущего запуска, где хранятся артефакты
                транскрипта.
            **kwargs (object): Аргументы конфигурации, передаваемые в ``BaseSTTAgent``.

        Returns:
            None: Агент сохраняет состояние сессии и модели.
        """
        self.session_dir = session_dir
        super().__init__(**kwargs)

    def run(self, audio_file_path: str | os.PathLike) -> Path:
        """Транскрибирует аудиофайл в текст.

        Args:
            audio_file_path (str | os.PathLike): Путь к аудиофайлу лекции,
                который открывает последовательный пайплайн.

        Returns:
            Path: Путь к сохраненному JSON-файлу транскрипции для локальной
            кластеризации.
        """
        if not os.path.exists(audio_file_path):
            logger.error(f"Файл не найден: {audio_file_path}")
            raise FileNotFoundError(f"Файла {audio_file_path}")

        logger.info(f"Начинаем обработку аудиофайла: {audio_file_path}...")

        segments, info = self.model.transcribe(
            audio=audio_file_path,
            initial_prompt=self.initial_prompt,
            **asdict(self._gen_config),
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
            logger.debug(
                f"Язык был задан явно, вероятности альтернатив недоступны: {self._gen_config.language}"
            )

        duration_mins = info.duration / 60
        logger.info(f"Исходная длительность аудио: {duration_mins:.2f} мин.")

        if info.duration_after_vad is not None:
            vad_mins = info.duration_after_vad / 60
            saved_mins = duration_mins - vad_mins
            logger.info(
                f"Длительность полезного сигнала (без тишины): {vad_mins:.2f} мин."
            )
            logger.info(f"VAD-фильтр вырезал {saved_mins:.2f} мин. тишины.")

        full_text_segments = []

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
                full_text_segments.append(text)

                progress = segment.end - pbar.n
                pbar.update(progress)

        full_text_dict = {"answer_agent": "\n".join(full_text_segments)}

        out_filepath = self._safe_result_out_line(
            output=full_text_dict,
            stage=self._app_config.name_stage_dir,
            file_name="out_filepath.json",
            session_dir=self.session_dir,
        )

        return out_filepath
