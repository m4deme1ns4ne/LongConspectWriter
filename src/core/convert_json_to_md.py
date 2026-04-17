import json
import os
import time


def convert_json_to_md(path, output_path="data/example-final-conspect"):
    with open(path, "r", encoding="utf-8") as file:
        conspect = json.load(file)

        final_conspect = ""

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
            f"finileconspect-{timestamp}.md",
        )

        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            f.write(final_conspect)

        return out_filepath
