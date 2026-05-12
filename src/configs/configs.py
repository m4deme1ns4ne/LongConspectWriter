"""Dataclass-конфигурации для пайплайна LongConspectWriter.

Эти dataclass описывают STT, LLM-агентов, кластеризаторы и полный пакет сессии,
который передается в строго последовательный оркестратор пайплайна.
"""

from dataclasses import dataclass
import torch
import os
from typing import Any
from pathlib import Path
from loguru import logger


@dataclass
class VadParametersConfig:
    """Настройки voice activity detection для предварительной обработки STT."""

    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 400
    threshold: float = 0.7


@dataclass
class STTInitConfig:
    """Настройки инициализации STT-модели FasterWhisper."""

    model_size_or_path: str
    device: str | None = None
    compute_type: str | None = None

    def __post_init__(self) -> None:
        """Заполняет устройство и тип вычислений по умолчанию для транскрибации.

        Изменяет поля ``device`` и ``compute_type`` экземпляра конфига на месте.
        """
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"


@dataclass
class STTGenConfig:
    """Настройки генерации для этапа транскрибации FasterWhisper."""

    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.45
    compression_ratio_threshold: float = 2.4
    language: str | None = None
    vad_parameters: VadParametersConfig | dict[str, int | float] | None = None

    def __post_init__(self) -> None:
        """Нормализует вложенные VAD-параметры, загруженные из YAML.

        Returns:
            None: VAD-настройки в виде dict преобразуются в ``VadParametersConfig``.
        """
        if isinstance(self.vad_parameters, dict):
            self.vad_parameters = VadParametersConfig(**self.vad_parameters)


@dataclass
class AppSTTConfig:
    """Настройки STT-агента, используемые со стороны пайплайна."""

    agent_name: str
    prompt_path: str | os.PathLike
    name_stage_dir: str


@dataclass
class LLMInitConfig:
    """Настройки инициализации агентов на llama.cpp."""

    model_path: str | None = None
    n_gpu_layers: int = -1
    n_ctx: int = 8192
    verbose: bool = False
    repo_id: str | None = None
    filename: str | None = None
    path_to_load_models: Path = Path(".models/")

    def __post_init__(self) -> None:
        """Проверяет источник модели и гарантирует наличие директории кеша моделей.

        Returns:
            None: Экземпляр конфига валидируется на месте.
        """
        if not self.path_to_load_models.exists():
            self.path_to_load_models.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"Папка {self.path_to_load_models} не была найдена. Поэтому она была создана автоматически."
            )

        if not (self.model_path or (self.repo_id and self.filename)):
            error_msg = (
                "Ошибка! Ни model_path, ни связка repo_id + filename не переданы."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)


@dataclass
class LLMGenConfig:
    """Параметры генерации, общие для агентов на llama.cpp."""

    max_tokens: int
    repeat_penalty: float = 1.1
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    min_p: float = 0.05


@dataclass
class LLMAppConfig:
    """Настройки LLM-агентов, используемые со стороны пайплайна."""

    agent_name: str
    prompt_path: str | os.PathLike
    name_stage_dir: str
    chunk_size_ratio: float | None = None
    chunk_overlap_ratio: float | None = None
    last_tail_words_count: int | None = None
    scheme_output_path: str | os.PathLike | None = None
    error_massage: str | None = None
    bad_code: str | None = None
    re_try_count: int | None = None
    step_temperature: float | None = None
    available_lib: str | None = None


@dataclass
class PipelineConfig:
    """Верхнеуровневые runtime-настройки пайплайна."""

    output_dir: str | os.PathLike
    lecture_theme: str = "universal"


@dataclass
class AgentConfigBundle:
    """Группирует все конфиги для одного агента.

    Пакет держит вместе init-, generation- и app-конфиги при передаче
    настроек этапа в оркестратор LongConspectWriter.
    """

    init_config: Any
    gen_config: Any
    app_config: Any


@dataclass
class PipelineSessionConfig:
    """Полный пакет конфигурации для одной сессии пайплайна."""

    pipeline: PipelineConfig
    stt: AgentConfigBundle
    synthesizer: AgentConfigBundle
    extractor: AgentConfigBundle
    local_planner: AgentConfigBundle
    global_planner: AgentConfigBundle
    local_clusterizer: AgentConfigBundle
    global_clusterizer: AgentConfigBundle
    grapher: AgentConfigBundle
    graph_planner: AgentConfigBundle


@dataclass
class LocalClusterizerInitConfig:
    """Настройки инициализации локальной семантической кластеризации."""

    model_name: str
    device: str = None


@dataclass
class LocalClusterizerGenConfig:
    """Параметры агломеративной кластеризации для локальных кластеров предложений."""

    threshold: float
    linkage: str
    turn_on_connectivity: bool
    metric: str
    n_clusters: bool | int


@dataclass
class GlobalClusterizerInitConfig:
    """Настройки инициализации глобального семантического распределения кластеров."""

    model_name: str
    device: str = "cpu"
