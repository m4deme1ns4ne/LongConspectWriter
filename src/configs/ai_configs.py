from dataclasses import dataclass
from os import PathLike
import torch


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
    output_dir: str | PathLike


@dataclass
class LlamaCppInitConfig:
    model_path: str
    n_gpu_layers: int = -1
    n_ctx: int = 8192
    verbose: bool = False


@dataclass
class LlamaCppGenConfig:
    max_tokens: int
    repeat_penalty: float = 1.1
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    min_p: float = 0.05


@dataclass
class AppLLMConfig:
    agent_name: str
    prompt_path: str | PathLike
    output_dir: str | PathLike
