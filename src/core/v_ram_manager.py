from loguru import logger
import gc
import torch


class VRamUsage:
    @staticmethod
    def get_vram_usage() -> str:
        if not torch.cuda.is_available():
            return "CPU"

        try:
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            allocated = torch.cuda.memory_allocated(device_index) / (1024**2)
            reserved = torch.cuda.memory_reserved(device_index) / (1024**2)
            total = props.total_memory / (1024**2)
            return (
                f"allocated={allocated:.0f} MB, "
                f"reserved={reserved:.0f} MB / {total:.0f} MB"
            )
        except Exception as exc:
            return f"GPU (usage unavailable: {exc})"


class VRamCleaner:
    @staticmethod
    def empty_vram(caller_name: str | None = None) -> None:
        owner = caller_name or "unknown"
        gc.collect()

        if not torch.cuda.is_available():
            logger.debug(f"[{owner}] GPU не использовался.")
            return

        try:
            before = VRamUsage.get_vram_usage()

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()

            torch.cuda.empty_cache()

            after = VRamUsage.get_vram_usage()
            logger.debug(f"[{owner}] VRAM очищена: {before} -> {after}")
        except Exception:
            logger.exception(f"[{owner}] Не удалось корректно очистить VRAM.")


# def decorator_v_ram_cleaner(func: Callable[..., object]) -> Callable[..., object]:
#     @functools.wraps(func)
#     def wrapper(*args: object, **kwargs: object) -> object:
#         name = func.__qualname__
#         start = time.perf_counter()

#         try:
#             result = func(*args, **kwargs)
#         except Exception:
#             elapsed = time.perf_counter() - start
#             logger.exception(f"[{name}] Упал через {elapsed:.1f} сек.")
#             raise
#         else:
#             elapsed = time.perf_counter() - start
#             logger.success(f"[{name}] Завершен за {elapsed:.1f} сек.")
#             return result
#         finally:
#             _VRamCleaner.empty_vram(caller_name=name)

#     return wrapper
