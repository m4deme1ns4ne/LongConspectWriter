from src.ai_configs import STTModelConfig, AIModelConfig
from src.transcribing import FasterWhisper


def main():
    stt_model_config = STTModelConfig(model_size= "large-v3",device="cpu", compute_type="int8")
    faster_whisper = FasterWhisper(stt_model_config)

    faster_whisper.transcribing(audio_file_path="data/example-audio/Защита информации. Лекция 1. (mp3cut.net).mp3")



if __name__ == "__main__":
    main()
