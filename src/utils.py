from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm
from transformers.generation.streamers import BaseStreamer


class TextsSplitter:
    @staticmethod
    def split_text(text: str, model_name: str, chunk_size: int = 2000) -> list[str]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        )

        chunks = splitter.split_text(text)
        logger.success(
            f"Текст успешно разбит на {len(chunks)} фрагментов по {chunk_size} токенов каждый."
        )
        return chunks


class TqdmTokenStreamer(BaseStreamer):
    def __init__(self, pbar: tqdm) -> None:
        self.pbar = pbar
        self.is_first = True

    def put(self, value: object) -> None:
        if self.is_first:
            self.is_first = False
            return

        self.pbar.update(1)

    def end(self) -> None:
        if self.pbar:
            self.pbar.close()


bad_words = [
    # ИИ-мусор и услужливость
    "конечно",
    "Конечно",
    "разумеется",
    "Разумеется",
    "вот ваш конспект",
    "Вот конспект",
    "я составил",
    "Я составил",
    "я подготовил",
    "Я подготовил",
    "представляю вам",
    "Представляю вам",
    "обратите внимание",
    "Обратите внимание",
    "давайте разберём",
    "Давайте разберём",
    # Существительные-нарраторы (ОБЯЗАТЕЛЬНО с большой и маленькой буквы)
    "лектор",
    "Лектор",
    "лектора",
    "Лектора",
    "лектором",
    "Лектором",
    "спикер",
    "Спикер",
    "спикера",
    "Спикера",
    "спикером",
    "Спикером",
    "докладчик",
    "Докладчик",
    "докладчика",
    "Докладчика",
    "докладчиком",
    "Докладчиком",
    "преподаватель",
    "Преподаватель",
    "преподавателя",
    "Преподавателя",
    "преподавателем",
    "Преподавателем",
    "профессор",
    "Профессор",
    "профессора",
    "Профессора",
    "профессором",
    "Профессором",
    "автор",
    "Автор",
    "автора",
    "Автора",
    "автором",
    "Автором",
    # Отсылки к формату лекции
    "в лекции",
    "В лекции",
    "в данной лекции",
    "В данной лекции",
    "в этой лекции",
    "В этой лекции",
    "в выступлении",
    "В выступлении",
    "в докладе",
    "В докладе",
    # Вводные конструкции пересказа
    "по словам",
    "По словам",
    "она отмечает",
    "Она отмечает",
    "он отмечает",
    "Он отмечает",
    "согласно лектору",
    "Согласно лектору",
    "в лекции говорится",
    "В лекции говорится",
    "он говорит",
    "Он говорит",
    "она говорит",
    "Она говорит",
]
