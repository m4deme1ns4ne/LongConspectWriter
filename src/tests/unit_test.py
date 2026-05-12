"""Unit-тесты для критических узлов LongConspectWriter.

Тестируется чистая логика без загрузки LLM/STT/embedding моделей.
object.__new__(Cls) создаёт экземпляр класса в обход __init__,
чтобы не запускать загрузку весов.

Запуск:
    # Все тесты
    uv run python -m pytest src/tests/unit_test.py -v

    # Одна группа
    uv run python -m pytest src/tests/unit_test.py::TestFormatClusterOutput -v

    # Один тест
    uv run python -m pytest src/tests/unit_test.py::TestFormatClusterOutput::test_small_tail_merges_into_previous_cluster -v

    # Остановиться на первом упавшем
    uv run python -m pytest src/tests/unit_test.py -v -x

    # Показать print() внутри тестов
    uv run python -m pytest src/tests/unit_test.py -v -s
"""

import json
import numpy as np
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# _format_cluster_output — алгоритм сглаживания хронологических кластеров
# ---------------------------------------------------------------------------

class TestFormatClusterOutput:
    """Тесты алгоритма сглаживания хронологических кластеров транскрипта.

    Метод _format_cluster_output принимает сырые метки AgglomerativeClustering
    и «склеивает» слишком маленькие чанки с соседними, чтобы синтезатор
    не получал кластер из одного-двух предложений.
    """

    def setup_method(self) -> None:
        """Создаёт экземпляр кластеризатора без вызова __init__.

        object.__new__ выделяет память под объект и пропускает __init__,
        поэтому sentence_transformers и CUDA не загружаются.
        Нам нужен только сам метод _format_cluster_output, который
        не обращается к self.model и не требует инициализированных атрибутов.
        """
        from src.core.clustering import SemanticLocalClusterizer
        self.obj = object.__new__(SemanticLocalClusterizer)

    def test_all_same_label_gives_one_cluster(self) -> None:
        """Если все предложения получили одну метку — должен быть ровно один кластер."""
        sentences = [f"Предложение {i}." for i in range(6)]
        labels = np.array([0] * 6)

        clusters, final_labels = self.obj._format_cluster_output(sentences, labels, min_sentences=5)

        assert len(clusters) == 1
        # final_labels должны совпадать по индексу с индексами clusters
        assert all(l == 0 for l in final_labels)

    def test_two_large_chunks_give_two_clusters(self) -> None:
        """Два больших блока с разными метками дают два отдельных кластера."""
        sentences = [f"Предложение {i}." for i in range(12)]
        labels = np.array([0] * 6 + [1] * 6)

        clusters, _ = self.obj._format_cluster_output(sentences, labels, min_sentences=5)

        assert len(clusters) == 2

    def test_small_tail_merges_into_previous_cluster(self) -> None:
        """Хвост меньше min_sentences слипается с предыдущим кластером.

        6 предложений с меткой 0, потом 2 с меткой 1.
        Хвост (2 < min_sentences=5) не набирает порог — прилипает к предыдущему.
        Итого: 1 кластер, а не 2.
        """
        sentences = [f"Предложение {i}." for i in range(8)]
        labels = np.array([0] * 6 + [1] * 2)

        clusters, _ = self.obj._format_cluster_output(sentences, labels, min_sentences=5)

        assert len(clusters) == 1
        # Убеждаемся, что хвостовые предложения физически вошли в кластер
        assert "Предложение 7." in clusters[0]

    def test_small_midstream_chunk_does_not_create_own_cluster(self) -> None:
        """Маленький блок в середине не триггерит сплит, а прилипает к следующему.

        Последовательность меток: [0]*6 + [1]*2 + [0]*6.
        При смене 0→1 текущий чанк (6 штук) закрывается и уходит в clusters.
        При смене 1→0 текущий чанк (2 штуки) < min_sentences — сплита нет,
        предложения добавляются в current_chunk дальше.
        Итого: 2 кластера ([0..5] и [6..13]), а не 3.
        """
        sentences = [f"Предложение {i}." for i in range(14)]
        labels = np.array([0] * 6 + [1] * 2 + [0] * 6)

        clusters, _ = self.obj._format_cluster_output(sentences, labels, min_sentences=5)

        assert len(clusters) == 2

    def test_cluster_text_contains_all_sentences(self) -> None:
        """Текст кластера — это join всех вошедших предложений через пробел."""
        sentences = ["Первое.", "Второе.", "Третье.", "Четвёртое.", "Пятое.", "Шестое."]
        labels = np.array([0] * 6)

        clusters, _ = self.obj._format_cluster_output(sentences, labels, min_sentences=5)

        assert "Первое." in clusters[0]
        assert "Шестое." in clusters[0]

    def test_final_labels_length_equals_sentences_count(self) -> None:
        """Длина final_labels должна совпадать с числом предложений на входе.

        Это инвариант: каждое предложение должно получить метку кластера,
        иначе визуализатор кластеров нарисует неправильный график.
        """
        sentences = [f"Предложение {i}." for i in range(10)]
        labels = np.array([0] * 5 + [1] * 5)

        _, final_labels = self.obj._format_cluster_output(sentences, labels, min_sentences=5)

        assert len(final_labels) == len(sentences)


# ---------------------------------------------------------------------------
# getting_graphs_from_conspect — парсер плейсхолдеров с вложенными скобками
# ---------------------------------------------------------------------------

class TestGettingGraphsFromConspect:
    """Тесты парсера плейсхолдеров [GRAPH_TYPE: ...] в тексте конспекта.

    Парсер реализован как счётчик скобок, поэтому поддерживает произвольную
    вложенность и не полагается на regex.
    Возвращает список кортежей (start, end, context), где context — это
    окно ±200 символов вокруг плейсхолдера для агента-граффера.
    """

    def setup_method(self) -> None:
        """Создаёт экземпляр пайплайна без вызова __init__.

        LongConspectWriterPipeline.__init__ запускает создание директории сессии
        и требует PipelineSessionConfig. object.__new__ обходит это —
        getting_graphs_from_conspect не использует self.* атрибуты.
        """
        from src.core.pipeline import LongConspectWriterPipeline
        self.obj = object.__new__(LongConspectWriterPipeline)

    def test_empty_string_returns_empty_list(self) -> None:
        """Пустая строка не содержит плейсхолдеров — результат пустой."""
        assert self.obj.getting_graphs_from_conspect("") == []

    def test_plain_text_without_placeholder_returns_empty(self) -> None:
        """Обычный текст без [GRAPH_TYPE: ...] — результат пустой."""
        assert self.obj.getting_graphs_from_conspect("Обычный текст лекции без графиков.") == []

    def test_single_placeholder_is_found(self) -> None:
        """Один плейсхолдер с вложенными скобками в данных корректно распознаётся."""
        text = "Здесь идёт [GRAPH_TYPE: bar, x=[1,2,3]] конец."

        result = self.obj.getting_graphs_from_conspect(text)

        assert len(result) == 1

    def test_placeholder_start_end_positions_are_bracket_chars(self) -> None:
        """start и end указывают ровно на символы '[' и ']' плейсхолдера."""
        text = "До [GRAPH_TYPE: line] после"

        result = self.obj.getting_graphs_from_conspect(text)
        start, end, _ = result[0]

        assert text[start] == "["
        assert text[end] == "]"

    def test_two_placeholders_both_found(self) -> None:
        """Два плейсхолдера в одном тексте — оба попадают в результат."""
        text = "[GRAPH_TYPE: bar, data=[1,2]] текст [GRAPH_TYPE: pie, data=[3,4]]"

        result = self.obj.getting_graphs_from_conspect(text)

        assert len(result) == 2

    def test_nested_brackets_do_not_break_parser(self) -> None:
        """Вложенные скобки внутри плейсхолдера не ломают счётчик.

        data=[[1,2],[3,4]] открывает два дополнительных уровня — парсер
        должен дождаться char_open_count == 0, а не завершиться на первом ']'.
        """
        text = "Текст [GRAPH_TYPE: scatter, data=[[1,2],[3,4]]] конец"

        result = self.obj.getting_graphs_from_conspect(text)

        assert len(result) == 1

    def test_context_window_includes_surrounding_text(self) -> None:
        """Третий элемент кортежа — контекстное окно ±200 символов вокруг плейсхолдера."""
        text = "Контекст до плейсхолдера. [GRAPH_TYPE: bar] Контекст после."

        _, _, context = self.obj.getting_graphs_from_conspect(text)[0]

        assert "Контекст до" in context
        assert "Контекст после" in context

    def test_unclosed_placeholder_is_not_returned(self) -> None:
        """Плейсхолдер без закрывающей скобки не должен попасть в результат.

        char_open_count никогда не вернётся к 0, поэтому append не вызывается.
        Парсер не должен падать — просто вернуть пустой список.
        """
        text = "Текст [GRAPH_TYPE: bar без закрытия"

        result = self.obj.getting_graphs_from_conspect(text)

        assert result == []


# ---------------------------------------------------------------------------
# TextsSplitter.split_text_to_sentences — разбиение транскрипта через razdel
# ---------------------------------------------------------------------------

class TestTextsSplitter:
    """Тесты разбиения транскрипта на предложения через razdel.

    split_text_to_sentences — это тонкая обёртка над razdel.sentenize.
    Тесты фиксируют поведение связки «наш код + библиотека»,
    чтобы при обновлении razdel сразу видеть, если семантика изменилась.
    """

    def test_normal_russian_text_splits_correctly(self) -> None:
        """Три предложения с разными знаками конца — три элемента в результате."""
        from src.core.utils import TextsSplitter
        text = "Это первое предложение. Это второе. А это третье!"

        result = TextsSplitter.split_text_to_sentences(text)

        assert len(result) == 3

    def test_single_sentence_returns_one_element(self) -> None:
        """Одно предложение — список из одного элемента с тем же текстом."""
        from src.core.utils import TextsSplitter
        text = "Одно единственное предложение."

        result = TextsSplitter.split_text_to_sentences(text)

        assert len(result) == 1
        assert result[0] == "Одно единственное предложение."

    def test_empty_string_returns_list_with_empty_element(self) -> None:
        """Пустая строка даёт [''], а не [] — это поведение razdel.sentenize.

        Зафиксировано как регрессионный тест: если razdel изменит поведение
        на возврат [], вниз по пайплайну это может сломать кластеризацию.
        """
        from src.core.utils import TextsSplitter

        result = TextsSplitter.split_text_to_sentences("")

        assert result == [""]

    def test_result_contains_only_strings(self) -> None:
        """Все элементы результата — строки, а не объекты razdel."""
        from src.core.utils import TextsSplitter
        text = "Первое предложение. Второе предложение."

        result = TextsSplitter.split_text_to_sentences(text)

        assert all(isinstance(s, str) for s in result)

    def test_sentences_preserve_original_text(self) -> None:
        """Специальные символы (формулы, степени) не теряются при сплите."""
        from src.core.utils import TextsSplitter
        text = "Формула Эйлера: e^(iπ) + 1 = 0. Это красиво."

        result = TextsSplitter.split_text_to_sentences(text)
        # Объединяем обратно, чтобы проверить сохранность символов
        full = " ".join(result)

        assert "Формула Эйлера" in full
        assert "e^(iπ)" in full


# ---------------------------------------------------------------------------
# convert_json_to_md — конвертация JSON-конспекта синтезатора в Markdown
# ---------------------------------------------------------------------------

class TestConvertJsonToMd:
    """Тесты конвертации JSON-конспекта синтезатора в Markdown-файл.

    convert_json_to_md читает JSON вида {тема: текст_или_список}
    и записывает Markdown с H1-заголовками и дисклеймером об AI.
    """

    def _make_pipeline(self, session_dir: Path) -> Any:
        """Создаёт экземпляр пайплайна и вручную задаёт session_dir.

        Без object.__new__ __init__ попытался бы создать папку сессии
        на диске и потребовал бы полный PipelineSessionConfig.
        actual_session_dir устанавливается вручную — convert_json_to_md
        использует только этот атрибут для записи выходного файла.

        Args:
            session_dir: Временная директория теста (обычно tmp_path из pytest).

        Returns:
            Экземпляр LongConspectWriterPipeline без инициализированного пайплайна.
        """
        from src.core.pipeline import LongConspectWriterPipeline
        obj = object.__new__(LongConspectWriterPipeline)
        obj.actual_session_dir = session_dir
        return obj

    def _write_json(self, tmp_path: Path, data: dict) -> Path:
        """Сериализует dict в JSON-файл внутри tmp_path и возвращает путь.

        Args:
            tmp_path: Временная директория, созданная pytest.
            data: Словарь конспекта {тема: текст}.

        Returns:
            Путь к записанному JSON-файлу.
        """
        p = tmp_path / "conspect.json"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return p

    def test_output_file_is_created_with_md_extension(self, tmp_path: Path) -> None:
        """Результирующий файл создаётся на диске и имеет расширение .md.

        tmp_path — встроенный pytest-фикстур: уникальная временная директория,
        которая автоматически удаляется после завершения теста.
        """
        json_path = self._write_json(tmp_path, {"Тема": "Текст."})

        result = self._make_pipeline(tmp_path).convert_json_to_md(json_path)

        assert result.exists()
        assert result.suffix == ".md"

    def test_topic_becomes_h1_header(self, tmp_path: Path) -> None:
        """Ключ словаря (тема) становится заголовком первого уровня # в Markdown."""
        json_path = self._write_json(tmp_path, {"Введение в интегралы": "Интеграл — площадь."})

        content = self._make_pipeline(tmp_path).convert_json_to_md(json_path).read_text(encoding="utf-8")

        assert "# Введение в интегралы" in content

    def test_string_body_is_present_in_output(self, tmp_path: Path) -> None:
        """Строковое тело темы дословно переносится в Markdown."""
        json_path = self._write_json(tmp_path, {"Тема": "Интеграл — площадь под кривой."})

        content = self._make_pipeline(tmp_path).convert_json_to_md(json_path).read_text(encoding="utf-8")

        assert "Интеграл — площадь под кривой." in content

    def test_list_body_all_items_present(self, tmp_path: Path) -> None:
        """Список в теле темы — все элементы попадают в Markdown через двойной перевод строки."""
        json_path = self._write_json(tmp_path, {"Тема": ["Первый абзац.", "Второй абзац."]})

        content = self._make_pipeline(tmp_path).convert_json_to_md(json_path).read_text(encoding="utf-8")

        assert "Первый абзац." in content
        assert "Второй абзац." in content

    def test_multiple_topics_all_present(self, tmp_path: Path) -> None:
        """Все темы из словаря присутствуют в Markdown в виде H1-заголовков."""
        data = {"Тема 1": "Текст 1.", "Тема 2": "Текст 2.", "Тема 3": "Текст 3."}
        json_path = self._write_json(tmp_path, data)

        content = self._make_pipeline(tmp_path).convert_json_to_md(json_path).read_text(encoding="utf-8")

        for topic in data:
            assert f"# {topic}" in content

    def test_ai_disclaimer_is_present(self, tmp_path: Path) -> None:
        """Дисклеймер об AI-генерации присутствует в начале каждого конспекта."""
        json_path = self._write_json(tmp_path, {"Тема": "Текст."})

        content = self._make_pipeline(tmp_path).convert_json_to_md(json_path).read_text(encoding="utf-8")

        assert "сгенерирован с помощью AI" in content


# ---------------------------------------------------------------------------
# BaseLlamaCppAgent._generate — распаковка ответа llama.cpp через мок модели
# ---------------------------------------------------------------------------

class TestGenerate:
    """Тесты метода _generate базового LLM-агента.

    Мокаем self.model.create_chat_completion — единственную точку контакта
    с llama.cpp — чтобы проверить логику распаковки ответа без загрузки весов.

    Используем AgentLocalPlanner как конкретный подкласс BaseLlamaCppAgent:
    BaseLlamaCppAgent сам по себе абстрактный (не реализует run),
    поэтому object.__new__(BaseLlamaCppAgent) выбросил бы TypeError.
    """

    def _make_agent(self) -> Any:
        """Создаёт AgentLocalPlanner с минимальным набором атрибутов для _generate.

        LLMGenConfig нужен настоящим датаклассом, а не MagicMock:
        _generate вызывает dataclasses.asdict(self._gen_config), который
        обходит __dict__ датакласса и упадёт на обычном объекте-заглушке.

        Returns:
            Экземпляр AgentLocalPlanner с мок-моделью и реальным gen_config.
        """
        from src.agents.agent_planner import AgentLocalPlanner
        from src.configs.configs import LLMGenConfig

        agent = object.__new__(AgentLocalPlanner)
        agent._gen_config = LLMGenConfig(max_tokens=100)
        agent.model = MagicMock()
        return agent

    def test_non_streaming_extracts_content_from_response(self) -> None:
        """Non-stream: content извлекается из choices[0]["message"]["content"]."""
        agent = self._make_agent()
        agent.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Ответ модели."}}]
        }

        result = agent._generate(
            prompt=[{"role": "user", "content": "Вопрос"}],
            stream=False,
        )

        assert result == "Ответ модели."

    def test_streaming_accumulates_all_chunks(self) -> None:
        """Stream: все delta.content конкатенируются в одну строку.

        Последний чанк без ключа "content" в delta — нормальная ситуация
        в протоколе llama.cpp, код должен его молча пропустить.
        """
        agent = self._make_agent()
        # iter() превращает список в генератор — именно так работает stream в llama.cpp
        agent.model.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "Привет"}}]},
            {"choices": [{"delta": {"content": ", "}}]},
            {"choices": [{"delta": {"content": "мир"}}]},
            {"choices": [{"delta": {}}]},  # последний чанк — нет "content"
        ])

        result = agent._generate(
            prompt=[{"role": "user", "content": "Вопрос"}],
            stream=True,
        )

        assert result == "Привет, мир"

    def test_non_streaming_passes_gen_config_to_model(self) -> None:
        """Параметры из LLMGenConfig попадают в вызов create_chat_completion.

        Проверяем что asdict() корректно разворачивает датакласс в kwargs.
        """
        agent = self._make_agent()
        agent.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        agent._generate(prompt=[{"role": "user", "content": "Q"}], stream=False)

        # call_args.kwargs содержит все именованные аргументы последнего вызова
        call_kwargs = agent.model.create_chat_completion.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# AgentLocalPlanner.run — фильтрация кластеров по метке [NO_TOPICS]
# ---------------------------------------------------------------------------

class TestLocalPlannerNoTopicsFiltering:
    """Тесты логики фильтрации [NO_TOPICS] в локальном планировщике.

    Если модель решает, что в кластере нет явных тем, она возвращает [NO_TOPICS].
    Такие кластеры не должны попадать в финальный план — иначе глобальный
    планировщик получит мусор и выдаст плохие заголовки глав.

    Мокаем _generate прямо на экземпляре (agent._generate = MagicMock(...)).
    Это работает потому что Python ищет атрибут сначала в __dict__ экземпляра,
    и наш MagicMock перекрывает метод класса до того, как тот дойдёт до tenacity.
    """

    def _make_planner(self, session_dir: Path) -> Any:
        """Создаёт AgentLocalPlanner со всеми атрибутами, нужными методу run.

        Args:
            session_dir: Временная директория, куда run запишет артефакт.

        Returns:
            Экземпляр AgentLocalPlanner без загруженной модели.
        """
        from src.agents.agent_planner import AgentLocalPlanner
        from src.configs.configs import LLMGenConfig, LLMAppConfig

        agent = object.__new__(AgentLocalPlanner)
        agent._gen_config = LLMGenConfig(max_tokens=100)
        agent._app_config = LLMAppConfig(
            agent_name="local_planner",
            prompt_path=".",
            name_stage_dir="test_local_plan",
        )
        agent.session_dir = session_dir
        # _build_prompt использует эти два атрибута для формирования промпта
        agent.system_prompt = "Ты помощник."
        agent.user_template = "Текст: {text}"
        agent.model = MagicMock()
        return agent

    def _write_clusters(self, tmp_path: Path, clusters: dict) -> Path:
        """Записывает словарь локальных кластеров в JSON-файл.

        Args:
            tmp_path: Временная директория теста.
            clusters: Словарь {id: текст_кластера}, как выдаёт SemanticLocalClusterizer.

        Returns:
            Путь к записанному JSON-файлу.
        """
        p = tmp_path / "local_clusters.json"
        p.write_text(json.dumps(clusters, ensure_ascii=False), encoding="utf-8")
        return p

    def test_normal_response_is_included_in_output(self, tmp_path: Path) -> None:
        """Ответ модели без [NO_TOPICS] попадает в финальный план."""
        json_path = self._write_clusters(tmp_path, {"0": "Текст про интегралы."})
        agent = self._make_planner(tmp_path)
        agent._generate = MagicMock(return_value="Микротема: Определение интеграла")

        result_path = type(agent)._orig_run(agent, json_path)

        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert "Определение интеграла" in result["answer_agent"]

    def test_no_topics_response_is_excluded_from_output(self, tmp_path: Path) -> None:
        """Ответ [NO_TOPICS] не попадает в финальный план — answer_agent пустой."""
        json_path = self._write_clusters(tmp_path, {"0": "Э-э-э, ну, значит..."})
        agent = self._make_planner(tmp_path)
        agent._generate = MagicMock(return_value="[NO_TOPICS]")

        result_path = type(agent)._orig_run(agent, json_path)

        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["answer_agent"] == ""

    def test_mixed_responses_only_valid_in_output(self, tmp_path: Path) -> None:
        """Из трёх кластеров один [NO_TOPICS] — в плане ровно два нормальных.

        side_effect задаёт разный возврат на каждый последующий вызов MagicMock:
        первый вызов → "Тема А", второй → "[NO_TOPICS]", третий → "Тема В".
        """
        clusters = {"0": "Кластер А.", "1": "Кластер Б.", "2": "Кластер В."}
        json_path = self._write_clusters(tmp_path, clusters)
        agent = self._make_planner(tmp_path)
        agent._generate = MagicMock(side_effect=["Тема А", "[NO_TOPICS]", "Тема В"])

        result_path = type(agent)._orig_run(agent, json_path)

        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert "Тема А" in result["answer_agent"]
        assert "[NO_TOPICS]" not in result["answer_agent"]
        assert "Тема В" in result["answer_agent"]


# ---------------------------------------------------------------------------
# _AgentExtractor.run — graceful degradation при сломанном JSON от модели
# ---------------------------------------------------------------------------

class TestExtractorJsonFallback:
    """Тесты обработки невалидного JSON-ответа в агенте-экстракторе.

    Экстрактор просит модель выдать список сущностей в JSON.
    Малые 8B-модели иногда ломают JSON (незакрытые скобки, лишний текст).
    В таком случае run() должен вернуть {"extracted_entities": []}
    вместо того чтобы упасть с JSONDecodeError и остановить весь пайплайн.
    """

    def _make_extractor(self, session_dir: Path) -> Any:
        """Создаёт _AgentExtractor без загрузки модели и схемы JSON.

        Args:
            session_dir: Временная директория для артефактов run().

        Returns:
            Экземпляр _AgentExtractor с мок-моделью и подставными конфигами.
        """
        from src.agents.agent_extractor import _AgentExtractor
        from src.configs.configs import LLMGenConfig, LLMAppConfig

        agent = object.__new__(_AgentExtractor)
        agent._gen_config = LLMGenConfig(max_tokens=100)
        agent._app_config = LLMAppConfig(
            agent_name="extractor",
            prompt_path=".",
            name_stage_dir="test_extractor",
        )
        agent.session_dir = session_dir
        agent.system_prompt = "Ты экстрактор сущностей."
        agent.user_template = "Текст: {text}"
        agent.model = MagicMock()
        # response_format передаётся в _generate, но _generate замокан — значение не важно
        agent.response_format = {"type": "json_object"}
        return agent

    def test_valid_json_response_is_parsed_correctly(self, tmp_path: Path) -> None:
        """Валидный JSON-ответ модели возвращается как распарсенный dict."""
        agent = self._make_extractor(tmp_path)
        agent._generate = MagicMock(
            return_value='{"extracted_entities": ["интегралы", "производная"]}'
        )

        result = type(agent)._orig_run(agent, synthesizer_chunk="Интегралы и производные.")

        assert result["extracted_entities"] == ["интегралы", "производная"]

    def test_invalid_json_returns_empty_entities_without_crash(self, tmp_path: Path) -> None:
        """Сломанный JSON не роняет пайплайн — возвращается пустой fallback.

        Пайплайн продолжает работу: синтез следующего чанка не прерывается,
        просто этот набор сущностей не попадает в already_seen_themes.
        """
        agent = self._make_extractor(tmp_path)
        agent._generate = MagicMock(return_value="не JSON {{{")

        result = type(agent)._orig_run(agent, synthesizer_chunk="Текст лекции.")

        assert result == {"extracted_entities": []}


# ---------------------------------------------------------------------------
# GlobalClusterVisualizer.run — регрессия ACCESS_VIOLATION (0xC0000005)
# ---------------------------------------------------------------------------

class TestGlobalClusterVisualizer:
    """Тесты визуализатора глобальных кластеров.

    Регрессия: run() падал с ACCESS_VIOLATION (exitcode=0xC0000005) внутри
    subprocess SemanticGlobalClusterizer из-за sns.scatterplot, который
    уходил в C-код FreeType при рендеринге текста в легенде.
    Фикс: заменён sns.scatterplot на plt.scatter.

    GlobalClusterVisualizer наследует BaseClusterVisualizer(Trackable), а НЕ Base,
    поэтому subprocess-изоляции нет — можно создавать напрямую без object.__new__
    и сразу вызывать run().
    """

    @staticmethod
    def _make_embeddings(n: int, dim: int = 8) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random((n, dim)).astype(np.float32)

    def test_run_does_not_crash(self, tmp_path: Path) -> None:
        """Базовый сценарий не должен падать с ACCESS_VIOLATION или любым исключением."""
        from src.core.clustering import GlobalClusterVisualizer

        viz = GlobalClusterVisualizer(tmp_path)
        viz.run(
            embeddings=self._make_embeddings(10),
            assignments=[0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
            chapter_titles=["Ch A", "Ch B", "Ch C"],
        )

    def test_run_saves_png(self, tmp_path: Path) -> None:
        """После run() в директории сессии появляется ровно один PNG."""
        from src.core.clustering import GlobalClusterVisualizer

        viz = GlobalClusterVisualizer(tmp_path)
        viz.run(
            embeddings=self._make_embeddings(6),
            assignments=[0, 0, 1, 1, 0, 1],
            chapter_titles=["Тема первая", "Тема вторая"],
        )

        pngs = list(tmp_path.rglob("*.png"))
        assert len(pngs) == 1, f"Ожидался один PNG, найдено: {pngs}"

    def test_run_skips_when_too_few_embeddings(self, tmp_path: Path) -> None:
        """Если embedding всего один — run() возвращается без создания файла."""
        from src.core.clustering import GlobalClusterVisualizer

        viz = GlobalClusterVisualizer(tmp_path)
        viz.run(
            embeddings=self._make_embeddings(1),
            assignments=[0],
            chapter_titles=["Единственная глава"],
        )

        assert list(tmp_path.rglob("*.png")) == []

    def test_run_with_cyrillic_chapter_titles(self, tmp_path: Path) -> None:
        """Кириллика в заголовках глав не вызывает падение — это была первопричина бага."""
        from src.core.clustering import GlobalClusterVisualizer

        viz = GlobalClusterVisualizer(tmp_path)
        viz.run(
            embeddings=self._make_embeddings(8),
            assignments=[0, 0, 0, 1, 1, 1, 0, 1],
            chapter_titles=[
                "Концепция программирования как искусства",
                "Архитектура программного обеспечения",
            ],
        )

    def test_run_more_chapters_than_tab10_colors(self, tmp_path: Path) -> None:
        """Больше 10 глав — цвета зациклены через % len(colors), падения нет."""
        from src.core.clustering import GlobalClusterVisualizer

        n_chapters = 12
        viz = GlobalClusterVisualizer(tmp_path)
        viz.run(
            embeddings=self._make_embeddings(n_chapters * 2),
            assignments=[i % n_chapters for i in range(n_chapters * 2)],
            chapter_titles=[f"Глава {i}" for i in range(n_chapters)],
        )
