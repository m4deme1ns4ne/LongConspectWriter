from dataclasses import dataclass


@dataclass
class AIModelConfig:
    """Родительский класс конфига для всех ИИ моделей"""
    pass
    
@dataclass
class LLMModelConfig(AIModelConfig):
    """Класс конфига для всех LLM моделей"""
    pass
    # Свои параметры для каждого LLM агента

@dataclass
class STTModelConfig(AIModelConfig):
    """Класс конфига для всех STT моделей"""
    model_size: str
    device: str
    compute_type: str
