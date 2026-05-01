"""Базовые абстракции для последовательного пайплайна LongConspectWriter.

Модуль задает общие контракты для агентов, кластеризаторов, оркестратора
пайплайна и визуализаторов. Классы сохраняют соглашение проекта: каждый этап
создает путь к артефакту и передает его следующему этапу.
"""

from abc import ABC, abstractmethod
from loguru import logger
from src.core.vram_manager import VRamCleaner
from src.core.utils import log_execution_time, LoadPrompts, modify_retry
from src.configs.configs import (
    LLMInitConfig,
    LLMGenConfig,
    LLMAppConfig,
    LocalClusterizerInitConfig,
    LocalClusterizerGenConfig,
    GlobalClusterizerInitConfig,
)
from dataclasses import asdict
from typing import Any
from types import TracebackType
from llama_cpp import Llama
from src.configs.configs import AppSTTConfig, STTGenConfig, STTInitConfig
from faster_whisper import WhisperModel
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import time
from dataclasses import asdict


class Trackable:
    """Mixin, который логирует время выполнения метода ``run`` у подкласса.

    The mixin is used by LongConspectWriter stages to observe sequential stage
    без изменения реализации этапа.
    """

    def __init_subclass__(cls: type["Trackable"], **kwargs: Any) -> None:
        """Оборачивает метод ``run`` подкласса логированием времени выполнения.

        Args:
            **kwargs (Any): Стандартные параметры инициализации подкласса.

        Returns:
            None: Подкласс изменяется на месте, если он определяет ``run``.

        Raises:
            Exception: Пробрасывает ошибки из ``object.__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            cls.run = log_execution_time(cls.run)


class Base(ABC):
    """Абстрактная база для каждого исполняемого этапа LongConspectWriter.

    Подклассы реализуют ``run`` и обычно возвращают путь к артефакту,
    который потребляет следующий строго упорядоченный этап пайплайна.
    """

    @abstractmethod
    def run(self) -> Any:
        """Выполняет этап пайплайна.

        Returns:
            Any: Результат конкретного этапа, обычно путь к следующему артефакту.

        Raises:
            NotImplementedError: Реализуется конкретными этапами пайплайна.
        """
        pass

    def _get_output_file_path(
        self, session_dir: Path, stage: str, file_name: str
    ) -> Path:
        """Строит и создает выходную директорию для артефакта этапа.

        Args:
            session_dir (Path): Директория текущей сессии пайплайна.
            stage (str): Имя поддиректории этапа внутри сессии.
            file_name (str): Имя файла артефакта для выхода этого этапа.

        Returns:
            Path: Полный путь, куда этап должен сохранить свой артефакт.

        Raises:
            OSError: Если директорию этапа невозможно создать.
        """
        actual_stage_path = session_dir / stage

        actual_stage_path.mkdir(parents=True, exist_ok=True)

        output_file_path = actual_stage_path / file_name

        return output_file_path

    def _safe_result_out_line(
        self,
        output: dict[str, Any] | str,
        stage: str,
        file_name: str,
        session_dir: Path,
        extension: str = "json",
        extension_file_writer: str = "w",
    ) -> Path:
        """Сохраняет результат этапа на диск и возвращает путь к артефакту.

        Args:
            output (dict[str, Any] | str): Выход этапа для сериализации в JSON или
                Markdown-текст для следующего шага пайплайна.
            stage (str): Поддиректория этапа внутри текущей сессии.
            file_name (str): Имя выходного артефакта.
            session_dir (Path): Директория текущей сессии пайплайна.
            extension (str): Режим сериализации, сейчас ``json`` или ``md``.
            extension_file_writer (str): Режим открытия файла для записи.

        Returns:
            Path: Путь сохраненного артефакта, передаваемый между этапами пайплайна.

        Raises:
            OSError: Если выходной файл невозможно записать.
            TypeError: Если JSON-выход невозможно сериализовать.
        """
        output_file_path = self._get_output_file_path(
            session_dir=session_dir, stage=stage, file_name=file_name
        )

        if extension == "json":
            with open(
                output_file_path, extension_file_writer, encoding="utf-8"
            ) as file:
                json.dump(output, file, ensure_ascii=False, indent=4)
        elif extension == "md":
            with open(output_file_path, extension_file_writer, encoding="utf-8") as f:
                f.write(str(output))
        else:
            logger.critical(
                f"Работа агента {self.__class__.__name__} не сохранена в файл по пути: {output_file_path}. Расширения {extension} нету в функции {self.__qualname__}"
            )
        logger.success(
            f"Работа агента {self.__class__.__name__} сохранена в файл по пути: {output_file_path}"
        )
        return output_file_path


class BaseAgent(Trackable, Base):
    """Базовый класс для агентов, владеющих моделью.

    The context-manager protocol releases model and tokenizer attributes after
    после завершения этапа агента, чтобы следующий этап мог переиспользовать GPU-память.
    """

    def __enter__(self) -> "BaseAgent":
        """Входит в контекст агента для одного последовательного этапа пайплайна.

        Returns:
            BaseAgent: Активный экземпляр агента.

        Raises:
            Exception: Намеренно не выбрасывает исключения.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Освобождает ресурсы принадлежащей агенту модели при выходе из этапа.

        Args:
            exc_type (type[BaseException] | None): Exception type raised inside
                контекста этапа, если оно было.
            exc_val (BaseException | None): Exception instance raised inside the
                stage context, if any.
            exc_tb (TracebackType | None): Traceback, возникший внутри
                контекста этапа, если он был.

        Returns:
            None: Очистка выполняется для моделей, которыми агент владеет.

        Raises:
            Exception: Ошибки очистки обрабатываются ``VRamCleaner``.
        """
        if getattr(self, "_owns_model", True):
            logger.debug(f"Очистка памяти от агента {self.__class__.__name__}...")
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            VRamCleaner.empty_vram(caller_name=self.__class__.__name__)


class BaseLLMAgent(BaseAgent):
    """Абстрактная база для LLM-агентов LongConspectWriter.

    Подклассы строят prompts из текущих артефактов пайплайна и генерируют текст
    для следующих этапов, не меняя протокол передачи контекста.
    """

    @abstractmethod
    def _generate(self, prompt: Any, **kwargs: Any) -> str:
        """Генерирует вывод модели для уже построенного prompt.

        Args:
            prompt (Any): Структура prompt, подготовленная из текущего артефакта
                пайплайна.
            **kwargs (Any): Agent-specific generation controls.

        Returns:
            str: Сгенерированный моделью текст.

        Raises:
            NotImplementedError: Реализуется конкретными LLM-агентами.
        """
        pass

    @abstractmethod
    def _build_prompt(self) -> Any:
        """Строит prompt модели из входа текущего этапа.

        Returns:
            Any: Структура prompt, ожидаемая конкретным backend модели.

        Raises:
            NotImplementedError: Реализуется конкретными LLM-агентами.
        """
        pass


class BaseLlamaCppAgent(BaseLLMAgent):
    """Базовый класс для агентов chat-completion на llama.cpp."""

    def __init__(
        self,
        init_config: LLMInitConfig,
        gen_config: LLMGenConfig,
        app_config: LLMAppConfig,
        lecture_theme: str,
        shared_model: Llama | None = None,
    ) -> None:
        """Инициализирует промпты и модель llama.cpp для агента пайплайна.

        Args:
            init_config (LLMInitConfig): Конфигурация загрузки модели для этого
                этапа агента.
            gen_config (LLMGenConfig): Generation parameters used by this
                agent stage.
            app_config (LLMAppConfig): Pipeline-facing agent configuration,
                including prompt paths and agent names.
            lecture_theme (str): Тема лекции, используемая для выбора system prompt.
            shared_model (Llama | None): Already loaded model reused by this
                агентом, когда владение должно остаться вне экземпляра.

        Returns:
            None: Инициализированный агент сохраняет состояние модели и prompt.

        Raises:
            OSError: Если локальную модель или файлы prompt невозможно загрузить.
            KeyError: Если отсутствуют обязательные ключи prompt.
        """
        super().__init__()
        self._init_config = init_config
        self._gen_config = gen_config
        self._app_config = app_config
        self.model_display_name = (
            self._init_config.model_path
            if self._init_config.model_path
            else f"{self._init_config.repo_id}/{self._init_config.filename}"
        )
        logger.info(
            f"Инициализация {self.__class__.__name__} (Модель: {self.model_display_name})"
        )

        if shared_model:
            self.model = shared_model
            self._owns_model = False
            logger.warning(
                f"Модель {self.model_display_name} уже была загружена в память."
            )
        else:
            init_kwargs = asdict(self._init_config)

            repo_id = init_kwargs.pop("repo_id", None)
            filename = init_kwargs.pop("filename", None)
            model_path = init_kwargs.pop("model_path")
            path_to_load_models = init_kwargs.pop("path_to_load_models", None)

            if repo_id and filename:
                logger.info(f"Загрузка модели с HuggingFace: {repo_id}/{filename}")
                self.model = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(path_to_load_models),
                    **init_kwargs,
                )
            else:
                logger.info(f"Загрузка локальной модели по пути: {model_path}")
                self.model = Llama(model_path=model_path, **init_kwargs)

            self._owns_model = True
            logger.info(f"Модель {self.model_display_name} загружена.")

        self.prompts = LoadPrompts.load_prompts(self._app_config.prompt_path)
        try:
            self.system_prompt = self.prompts[self._app_config.agent_name][
                "system_prompt"
            ][lecture_theme]
        except KeyError:
            logger.warning(
                f"Промпта для темы {lecture_theme} нету. Будет использован стандартный промпт 'universal'"
            )
            self.system_prompt = self.prompts[self._app_config.agent_name][
                "system_prompt"
            ]["universal"]

        logger.info(
            f"Загружен промпт для агента {self.__class__.__name__}, по тематике: {lecture_theme}"
        )
        self.user_template = self.prompts[self._app_config.agent_name]["user_template"]

        logger.debug(
            f"Параметры запуска агента {self.__class__.__name__}: {self._init_config}"
        )
        logger.debug(
            f"Параметры генерации агента {self.__class__.__name__}: {self._gen_config}"
        )
        logger.debug(
            f"Параметры использования агента {self.__class__.__name__}: {self._app_config}"
        )

    def _build_prompt(
        self, tokenizer: Any = None, **kwargs: Any
    ) -> list[dict[str, str]]:
        """Строит chat prompt llama.cpp из активного пользовательского шаблона.

        Args:
            tokenizer (Any): Опциональный токенизатор, используемый только для логирования длины prompt.
            **kwargs (Any): Значения, вставляемые в пользовательский шаблон из
                текущего артефакта пайплайна.

        Returns:
            list[dict[str, str]]: Chat-сообщения с ролями system и user.

        Raises:
            KeyError: Если обязательный ключ шаблона отсутствует в ``kwargs``.
        """
        user_prompt = self.user_template.format(**kwargs)
        if tokenizer is not None:
            len_tokenizer_system_prompt = len(
                tokenizer(self.system_prompt.encode("utf-8"))
            )
            len_tokenizer_user_prompt = len(tokenizer(user_prompt.encode("utf-8")))
            logger.debug(
                f"Текущая длинна общая длинна контекста: {len_tokenizer_system_prompt + len_tokenizer_user_prompt}"
            )
        return self._format_output(user_prompt)

    def _format_output(self, user_prompt: str) -> list[dict[str, str]]:
        """Форматирует пользовательский prompt в chat-сообщения llama.cpp.

        Args:
            user_prompt (str): Тело user-сообщения, сгенерированное для текущего
                этапа пайплайна.

        Returns:
            list[dict[str, str]]: Two-message chat prompt for llama.cpp.

        Raises:
            Exception: Намеренно не выбрасывает исключения.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @modify_retry
    def _generate(
        self,
        prompt: list[dict[str, str]],
        stream: bool = False,
        token_pbar: Any = None,
        logit_bias: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Генерирует текст из chat prompt llama.cpp.

        Args:
            prompt (list[dict[str, str]]): Chat-сообщения, построенные для текущего
                этапа агента.
            stream (bool): Нужно ли потреблять потоковые чанки модели.
            token_pbar (Any): Опциональный progress bar, обновляемый на каждый потоковый токен.
            logit_bias (dict[str, Any] | None): Опциональный logit bias для llama.cpp.
            response_format (dict[str, Any] | None): Опциональный запрос structured-output,
                передаваемый в llama.cpp.

        Returns:
            str: Сгенерированное содержимое assistant.

        Raises:
            Exception: Пробрасывает ошибки генерации модели после retry-попыток.
        """

        if logit_bias is None:
            logit_bias = {}

        kwargs = {
            "messages": prompt,
            "stream": stream,
            "logit_bias": logit_bias,
            **asdict(self._gen_config),
        }

        if response_format is not None:
            kwargs["response_format"] = response_format

        if stream:
            response_text = ""
            generator = self.model.create_chat_completion(**kwargs)

            for chunk in generator:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    text_chunk = delta["content"]
                    response_text += text_chunk
                    if token_pbar:
                        token_pbar.update(1)
            return response_text
        else:
            response = self.model.create_chat_completion(**kwargs)
            return response["choices"][0]["message"]["content"]


class BaseSTTAgent(BaseAgent):
    """Базовый класс для speech-to-text агентов первого этапа пайплайна."""

    def __init__(
        self,
        init_config: STTInitConfig,
        gen_config: STTGenConfig,
        app_config: AppSTTConfig,
        lecture_theme: str,
    ) -> None:
        """Загружает STT-модель и выбирает начальный prompt транскрибации.

        Args:
            init_config (STTInitConfig): Настройки инициализации модели Whisper.
            gen_config (STTGenConfig): Настройки генерации транскрибации.
            app_config (AppSTTConfig): STT-конфигурация со стороны пайплайна.
            lecture_theme (str): Тема лекции, используемая для выбора STT prompt.

        Returns:
            None: Инициализированный агент сохраняет состояние модели и prompt.

        Raises:
            OSError: Если модель или файлы prompt невозможно загрузить.
            KeyError: Если отсутствуют обязательные секции prompt.
        """
        super().__init__()
        self._init_config = init_config
        self._gen_config = gen_config
        self._app_config = app_config
        logger.info(
            f"Инициализация агента STT (Модель: {self._init_config.model_size_or_path}, Устройство: {self._init_config.device})"
        )
        logger.info(f"Загрузка {self._init_config.model_size_or_path} в память...")
        self.model = WhisperModel(
            **asdict(self._init_config),
        )
        logger.info(f"Модель {self._init_config.model_size_or_path} загружена.")

        logger.debug(
            f"Параметры запуска агента {self.__class__.__name__}: {self._init_config}"
        )
        logger.debug(
            f"Параметры генерации агента {self.__class__.__name__}: {self._gen_config}"
        )
        logger.debug(
            f"Параметры использования агента {self.__class__.__name__}: {self._app_config}"
        )
        self.prompt = LoadPrompts.load_prompts(self._app_config.prompt_path)

        try:
            self.initial_prompt = self.prompt[self._app_config.agent_name][
                lecture_theme
            ]
        except KeyError:
            logger.warning(
                f"Промпта для темы {lecture_theme} нету. Будет использован стандартный промпт 'universal'"
            )
            self.initial_prompt = self.prompt[self._app_config.agent_name]["universal"]
        logger.info(
            f"Загружен промпт для агента {self.__class__.__name__}, по тематике: {lecture_theme}"
        )


class BaseLocalClusterizer(Trackable, Base):
    """Базовый класс для этапов локальной хронологической кластеризации."""

    def __init__(
        self,
        init_config: LocalClusterizerInitConfig,
        gen_config: LocalClusterizerGenConfig,
    ) -> None:
        """Сохраняет конфигурацию локального кластеризатора.

        Args:
            init_config (LocalClusterizerInitConfig): Настройки embedding-модели
                для локальной кластеризации.
            gen_config (LocalClusterizerGenConfig): Local clustering parameters.

        Returns:
            None: Конфигурация сохраняется для конкретных кластеризаторов.

        Raises:
            Exception: Намеренно не выбрасывает исключения.
        """
        super().__init__()
        self._init_config = init_config
        self._gen_config = gen_config


class BaseGlobalClusterizer(Trackable, Base):
    """Базовый класс для сопоставления локальных кластеров с глобальными главами лекции."""

    def __init__(
        self,
        init_config: GlobalClusterizerInitConfig,
    ) -> None:
        """Сохраняет конфигурацию глобального кластеризатора.

        Args:
            init_config (GlobalClusterizerInitConfig): Embedding model settings
                используемой для выравнивания локальных кластеров с глобальным планом.

        Returns:
            None: Конфигурация сохраняется для конкретных кластеризаторов.

        Raises:
            Exception: Намеренно не выбрасывает исключения.
        """
        super().__init__()
        self._init_config = init_config


class BasePipeline(Trackable, Base):
    """Базовый класс полной сессии пайплайна LongConspectWriter."""

    def __post_init__(self) -> None:
        """Создает выходную директорию запуска с timestamp.

        Returns:
            None: Директория активной сессии сохраняется в экземпляре.

        Raises:
            OSError: Если директорию запуска невозможно создать.
        """

        output_dir = Path(self.pipeline_config.output_dir)
        now = datetime.now()

        self.actual_session_dir = (
            output_dir / "runs" / now.strftime("%Y.%m.%d") / now.strftime("%H.%M.%S")
        )
        self.actual_session_dir.mkdir(exist_ok=False, parents=True)

        logger.success(
            f"Папка актуальной сессии создана по пути: {self.actual_session_dir}"
        )


class BaseClusterVisualizer(Trackable):
    """Базовый класс для сохранения диагностических графиков кластеров в дерево сессии."""

    def __init__(self, session_dir: Path) -> None:
        """Инициализирует визуализатор для текущей сессии пайплайна.

        Args:
            session_dir (Path): Директория текущей сессии LongConspectWriter.

        Returns:
            None: Визуализатор сохраняет директорию сессии.

        Raises:
            Exception: Намеренно не выбрасывает исключения.
        """
        self.session_dir = session_dir

    def _add_watermark(self, metadata: dict[str, Any] | None) -> None:
        """Приватный метод для добавления текста метаданных на холст.

        Args:
            metadata (dict[str, Any] | None): Метаданные пайплайна и модели для
                вывода внизу диагностического графика.

        Returns:
            None: Активная фигура matplotlib изменяется на месте.

        Raises:
            Exception: Пробрасывает ошибки рендера matplotlib.
        """
        if not metadata:
            return
        info_str = " | ".join([f"{k}: {v}" for k, v in metadata.items()])
        plt.figtext(
            0.01, 0.01, info_str, fontsize=8, color="gray", alpha=0.7, ha="left"
        )

    def _save_and_close(
        self, sub_dir: str, prefix: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Общая логика рендера, сохранения файла и очистки памяти.

        Args:
            sub_dir (str): Поддиректория сессии, куда сохраняется график.
            prefix (str): Префикс имени файла графика, идентифицирующий этап.
            metadata (dict[str, Any] | None): Опциональные метаданные модели и этапа,
                добавляемые в watermark графика.

        Returns:
            None: График записывается на диск и закрывается.

        Raises:
            OSError: Если директорию или файл графика невозможно записать.
        """
        self._add_watermark(metadata)

        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.png"

        out_path = self.session_dir / sub_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.success(f"График {prefix} сохранен: {out_path}")
