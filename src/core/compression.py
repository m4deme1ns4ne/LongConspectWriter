import json
from transformers import AutoTokenizer
from src.core.vram_manager import VRamUsage


class SmartCompressor:
    def __init__(self, synthesizer_init_config, synthesizer_gen_config):
        self.synthesizer_init_config = synthesizer_init_config
        self.synthesizer_gen_config = synthesizer_gen_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.synthesizer_init_config.pretrained_model_name_or_path
        )

    def _compress_cluster(self): ...

    def process(self, path):
        # with open(path, "r", encoding="utf-8") as file:
        #     global_clusters = json.load(file)

        # allocated, reserved, total = VRamUsage.get_vram_usage()
        # for topik, clusters in global_clusters.items():
        #     token_count = len(self.tokenizer.encode(clusters, add_special_tokens=False))

        #     if self.synthesizer_init_config.pretrained_model_name_or_path + token_count > total:
        #
        ...
