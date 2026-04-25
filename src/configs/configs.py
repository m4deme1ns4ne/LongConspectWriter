from dataclasses import dataclass
import torch
import os
from typing import Any


@dataclass
class VadParametersConfig:
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 400
    threshold: float = 0.7


@dataclass
class STTInitConfig:
    model_size_or_path: str
    device: str | None = None
    compute_type: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"


@dataclass
class STTGenConfig:
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.45
    compression_ratio_threshold: float = 2.4
    language: str | None = None
    vad_parameters: VadParametersConfig | dict[str, int | float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.vad_parameters, dict):
            self.vad_parameters = VadParametersConfig(**self.vad_parameters)


@dataclass
class AppSTTConfig:
    agent_name: str
    prompt_path: str | os.PathLike


@dataclass
class LLMInitConfig:
    model_path: str
    n_gpu_layers: int = -1
    n_ctx: int = 8192
    verbose: bool = False


@dataclass
class LLMGenConfig:
    max_tokens: int
    repeat_penalty: float = 1.1
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    min_p: float = 0.05


@dataclass
class LLMAppConfig:
    agent_name: str
    prompt_path: str | os.PathLike
    chunk_size_ratio: float | None = None
    chunk_overlap_ratio: float | None = None
    last_tail_words_count: int = None
    scheme_output_path: dict = None


@dataclass
class PipelineConfig:
    output_dir: str | os.PathLike
    lecture_theme: str = "universal"


@dataclass
class AgentConfigBundle:
    """Группирует все конфиги для одного агента."""

    init_config: Any
    gen_config: Any
    app_config: Any


@dataclass
class PipelineSessionConfig:
    pipeline: PipelineConfig
    stt: AgentConfigBundle
    synthesizer: AgentConfigBundle
    extractor: AgentConfigBundle
    local_planner: AgentConfigBundle
    global_planner: AgentConfigBundle
    local_clusterizer: AgentConfigBundle
    global_clusterizer: AgentConfigBundle


@dataclass
class LocalClusterizerInitConfig:
    model_name: str


@dataclass
class LocalClusterizerGenConfig:
    threshold: float
    linkage: str
    turn_on_connectivity: bool
    metric: str
    n_clusters: bool | int


@dataclass
class GlobalClusterizerInitConfig:
    model_name: str
