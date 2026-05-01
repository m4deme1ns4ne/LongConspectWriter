"""Утилиты для наблюдения за GPU-памятью и ее очистки между этапами пайплайна.

Модуль используется агентами LongConspectWriter при завершении этапа, когда
загруженную модель можно освободить без изменения состояния последовательного
пайплайна.
"""

from typing import Optional

from loguru import logger
import gc
import torch
import pynvml


class VRamUsage:
    """Читает текущие счетчики VRAM для активного CUDA-устройства."""

    @staticmethod
    def get_vram_usage() -> tuple[float, float, float]:
        """Возвращает использованную, свободную и общую видеопамять (VRAM) устройства 0 в мегабайтах.

        Returns:
            tuple[float, float, float]: Значения использованной, свободной и общей VRAM в МБ.

        Raises:
            pynvml.NVMLError: Обрабатывается внутри функции и представляется в виде нулевых значений.
        """
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            used = info.used / (1024**2)
            total = info.total / (1024**2)
            free = info.free / (1024**2)

            pynvml.nvmlShutdown()
            return used, free, total
        except pynvml.NVMLError as exc:
            logger.warning(f"Ошибка NVML: {exc}")
            return 0, 0, 0


class VRamCleaner:
    """Очищает память Python и CUDA после завершения этапа агентом."""

    @staticmethod
    def empty_vram(caller_name: Optional[str] = None) -> None:
        """Освобождает кешированную CUDA-память для только что завершившего агента.

        Args:
            caller_name (Optional[str]): Имя агента LongConspectWriter или
                этапа, запросившего очистку. Используется только для диагностического логирования.

        Returns:
            None: Метод выполняет очистку и пишет логи.

        Raises:
            Exception: Обрабатывается внутри, потому что очистка не должна прерывать
                последовательный пайплайн после того, как этап создал выход.
        """
        owner = caller_name or "unknown"
        gc.collect()

        if not torch.cuda.is_available():
            logger.debug(f"[{owner}] GPU не использовался.")
            return

        try:
            before_allocated, before_reserved, before_total = VRamUsage.get_vram_usage()

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()

            torch.cuda.empty_cache()

            after_allocated, after_reserved, after_total = VRamUsage.get_vram_usage()
            logger.debug(
                f"[{owner}] VRAM очищена: allocated={before_allocated:.0f} MB, reserved={before_reserved:.0f} MB / {before_total:.0f} MB -> allocated={after_allocated:.0f} MB, reserved={after_reserved:.0f} MB / {after_total:.0f} MB"
            )
        except Exception:
            logger.exception(f"[{owner}] Не удалось корректно очистить VRAM.")
