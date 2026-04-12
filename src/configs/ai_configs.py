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
class LLMInitConfig:
    pretrained_model_name_or_path: str
    torch_dtype: torch.dtype | None = None
    device_map: dict[str, int] | str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    quantization_config: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.device_map is None:
            self.device_map = {"": 0} if torch.cuda.is_available() else "cpu"

        if self.torch_dtype is None:
            self.torch_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

        if isinstance(self.quantization_config, dict):
            self.quantization_config["bnb_4bit_compute_dtype"] = self.torch_dtype


@dataclass
class LLMGenConfig:
    max_new_tokens: int
    repetition_penalty: float
    temperature: float
    top_p: float
    do_sample: bool
    guidance_scale: float | None = None


@dataclass
class AppLLMConfig:
    agent_name: str
    prompt_path: str | PathLike
    output_dir: str | PathLike
