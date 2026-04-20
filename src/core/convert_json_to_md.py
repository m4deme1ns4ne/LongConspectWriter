import json
import os
import time
from loguru import logger


def convert_json_to_md(path, output_path="data/example-final-conspect"):
    with open(path, "r", encoding="utf-8") as file:
        conspect = json.load(file)

        final_conspect = """**Этот конспект сгенерирован мультиагентной системой на базе AI.**\n**Система может допускать ошибки в формулах, вычислениях и специфической терминологии.**\n**Пожалуйста, относитесь с понимаем и проверяйте конспект!**\n\n"""

        for topik, body in conspect.items():
            final_conspect += f"# {topik}\n\n"
            if isinstance(body, list):
                text_body = "\n\n".join(str(item) for item in body)
            else:
                text_body = str(body)
            final_conspect += f"{text_body}\n\n"

        timestamp = int(time.time())
        out_filepath = os.path.join(
            output_path,
            f"finalconspect-{timestamp}.md",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            f.write(final_conspect)

        logger.success(
            f"Итоговый отформатированный конспект успешно сохранен по пути: {out_filepath}"
        )
        return out_filepath


# convert_json_to_md("data/example-conspect\T-lite-it-2_1-Q5_K_M_gguf-intfloat_multilingual-e5-small-1776631396.json-1776633157.json")
