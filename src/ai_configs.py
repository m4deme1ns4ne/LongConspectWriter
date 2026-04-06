from dataclasses import dataclass, fields
import torch
from pathlib import Path
from os import PathLike


def _non_none_dataclass_fields(
    instance: object, exclude: set[str] | None = None
) -> dict[str, object]:
    ignored_fields = exclude or set()
    kwargs: dict[str, object] = {}

    for field in fields(instance):
        if field.name in ignored_fields:
            continue

        value = getattr(instance, field.name)
        if value is not None:
            kwargs[field.name] = value

    return kwargs


@dataclass
class LLMInitConfig:
    """Конфиг только для загрузки весов в память"""

    model: str
    agent_name: str
    torch_dtype: torch.dtype | None = None
    device_map: dict[str, int] | str | None = None
    prompt: Path | None = None
    trust_remote_code: bool | None = None
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        if self.device_map is None:
            self.device_map = {"": 0} if torch.cuda.is_available() else "cpu"

        if self.torch_dtype is None:
            self.torch_dtype = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

    def model_kwargs(self) -> dict[str, object]:
        return _non_none_dataclass_fields(
            self, exclude={"model", "agent_name", "prompt"}
        )


@dataclass
class LLMGenConfig:
    """Конфиг только для параметров генерации (kwargs)"""

    max_new_tokens: int | None = None
    repetition_penalty: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    negative_prompt_ids: torch.Tensor | None = None
    negative_prompt_attention_mask: torch.Tensor | None = None
    guidance_scale: float | None = None
    bad_words_ids: list[list[int]] | None = None

    def generation_kwargs(self, exclude: set[str] | None = None) -> dict[str, object]:
        return _non_none_dataclass_fields(self, exclude=exclude)


@dataclass
class VadParametersConfig:
    min_silence_duration_ms: int | None = None
    speech_pad_ms: int | None = None
    threshold: float | None = None

    def to_kwargs(self) -> dict[str, object]:
        return _non_none_dataclass_fields(self)


@dataclass
class STTInitConfig:
    """Класс конфига для всех STT моделей"""

    model_size_or_path: str
    output_dir: str | PathLike
    device: str | None = None
    compute_type: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"

    def model_kwargs(self) -> dict[str, object]:
        return _non_none_dataclass_fields(
            self, exclude={"model_size_or_path", "output_dir"}
        )


@dataclass
class STTGenConfig:
    beam_size: int | None = None
    vad_filter: bool | None = None
    condition_on_previous_text: bool | None = None
    no_speech_threshold: float | None = None
    compression_ratio_threshold: float | None = None
    language: str | None = None
    vad_parameters_config: VadParametersConfig | None = None

    def transcribe_kwargs(self) -> dict[str, object]:
        kwargs = _non_none_dataclass_fields(self, exclude={"vad_parameters_config"})

        if self.vad_parameters_config is not None:
            kwargs["vad_parameters"] = self.vad_parameters_config.to_kwargs()

        return kwargs
