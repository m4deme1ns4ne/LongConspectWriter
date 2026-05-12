# Тесты LongConspectWriter

## Как запускать

```bash
uv run python -m pytest src/tests/unit_test.py -v
```

### Дополнительные флаги

```bash
# Остановиться на первом упавшем тесте
uv run python -m pytest src/tests/unit_test.py -v -x

# Запустить одну группу тестов
uv run python -m pytest src/tests/unit_test.py::TestFormatClusterOutput -v

# Запустить один конкретный тест
uv run python -m pytest src/tests/unit_test.py::TestExtractorJsonFallback::test_invalid_json_returns_empty_entities_without_crash -v
```

---

## Описание тестов

### TestFormatClusterOutput — алгоритм сглаживания кластеров

`SemanticLocalClusterizer._format_cluster_output` принимает сырые метки
`AgglomerativeClustering` и «склеивает» слишком маленькие чанки с соседними,
чтобы синтезатор не получал кластер из одного-двух предложений.

| Тест | Что проверяет |
|---|---|
| `test_all_same_label_gives_one_cluster` | Если все предложения одного кластера — выход содержит ровно 1 кластер |
| `test_two_large_chunks_give_two_clusters` | Два больших блока с разными метками дают два кластера |
| `test_small_tail_merges_into_previous_cluster` | Хвост меньше `min_sentences` прилипает к предыдущему кластеру, а не создаёт свой |
| `test_small_midstream_chunk_does_not_create_own_cluster` | Маленький блок в середине не триггерит сплит — результат 2 кластера, не 3 |
| `test_cluster_text_contains_all_sentences` | Текст кластера содержит все вошедшие предложения |
| `test_final_labels_length_equals_sentences_count` | Длина `final_labels` равна числу предложений на входе |

---

### TestGettingGraphsFromConspect — парсер плейсхолдеров графиков

`LongConspectWriterPipeline.getting_graphs_from_conspect` ищет плейсхолдеры
`[GRAPH_TYPE: ...]` в тексте конспекта. Парсер работает как счётчик скобок
и поддерживает произвольную вложенность.

| Тест | Что проверяет |
|---|---|
| `test_empty_string_returns_empty_list` | Пустая строка → пустой список |
| `test_plain_text_without_placeholder_returns_empty` | Текст без плейсхолдеров → пустой список |
| `test_single_placeholder_is_found` | Один плейсхолдер найден |
| `test_placeholder_start_end_positions_are_bracket_chars` | `start` и `end` указывают ровно на `[` и `]` |
| `test_two_placeholders_both_found` | Два плейсхолдера — оба в результате |
| `test_nested_brackets_do_not_break_parser` | `data=[[1,2],[3,4]]` не ломает счётчик скобок |
| `test_context_window_includes_surrounding_text` | Третий элемент кортежа содержит ±200 символов вокруг плейсхолдера |
| `test_unclosed_placeholder_is_not_returned` | Незакрытый `[GRAPH_TYPE:` не падает и не возвращается |

---

### TestTextsSplitter — разбиение транскрипта на предложения

`TextsSplitter.split_text_to_sentences` — тонкая обёртка над `razdel.sentenize`.
Тесты фиксируют поведение связки «наш код + библиотека».

| Тест | Что проверяет |
|---|---|
| `test_normal_russian_text_splits_correctly` | 3 предложения с разными знаками конца → 3 элемента |
| `test_single_sentence_returns_one_element` | Одно предложение → список из одного элемента с тем же текстом |
| `test_empty_string_returns_list_with_empty_element` | Пустая строка → `['']` (поведение `razdel`, зафиксировано как регрессионный тест) |
| `test_result_contains_only_strings` | Все элементы — строки, а не объекты `razdel` |
| `test_sentences_preserve_original_text` | Формулы и спецсимволы не теряются при разбиении |

---

### TestConvertJsonToMd — конвертация конспекта JSON → Markdown

`LongConspectWriterPipeline.convert_json_to_md` читает JSON вида
`{тема: текст_или_список}` и записывает Markdown с H1-заголовками и дисклеймером.

| Тест | Что проверяет |
|---|---|
| `test_output_file_is_created_with_md_extension` | Файл создаётся на диске с расширением `.md` |
| `test_topic_becomes_h1_header` | Ключ словаря становится заголовком `# Тема` |
| `test_string_body_is_present_in_output` | Строковое тело темы дословно попадает в файл |
| `test_list_body_all_items_present` | Список в теле — все элементы присутствуют |
| `test_multiple_topics_all_present` | Все темы из словаря присутствуют в виде H1 |
| `test_ai_disclaimer_is_present` | Дисклеймер «сгенерирован с помощью AI» есть в каждом конспекте |

---

### TestGenerate — распаковка ответа llama.cpp (мок)

`BaseLlamaCppAgent._generate` — единственная точка контакта с llama.cpp.
`self.model.create_chat_completion` заменяется `MagicMock`, реальная модель не загружается.

| Тест | Что проверяет |
|---|---|
| `test_non_streaming_extracts_content_from_response` | Non-stream: `content` извлекается из `choices[0]["message"]["content"]` |
| `test_streaming_accumulates_all_chunks` | Stream: все `delta.content` конкатенируются, чанк без `content` молча пропускается |
| `test_non_streaming_passes_gen_config_to_model` | Параметры из `LLMGenConfig` (через `asdict`) реально попадают в вызов модели |

---

### TestLocalPlannerNoTopicsFiltering — фильтрация [NO_TOPICS] (мок)

`AgentLocalPlanner.run` фильтрует кластеры, где модель не нашла тем.
`_generate` заменяется `MagicMock` с нужными возвратами.

| Тест | Что проверяет |
|---|---|
| `test_normal_response_is_included_in_output` | Ответ без `[NO_TOPICS]` попадает в финальный план |
| `test_no_topics_response_is_excluded_from_output` | Ответ `[NO_TOPICS]` → `answer_agent` пустой |
| `test_mixed_responses_only_valid_in_output` | Из 3 кластеров 1 с `[NO_TOPICS]` → в плане ровно 2 нормальных |

---

### TestExtractorJsonFallback — устойчивость к сломанному JSON (мок)

`_AgentExtractor.run` просит модель выдать список сущностей в JSON.
Малые 8B-модели иногда ломают JSON — агент не должен падать.

| Тест | Что проверяет |
|---|---|
| `test_valid_json_response_is_parsed_correctly` | Валидный JSON → распарсенный `dict` с `extracted_entities` |
| `test_invalid_json_returns_empty_entities_without_crash` | Сломанный JSON → `{"extracted_entities": []}` без `JSONDecodeError` |

---

## Как воспринимать результат

### Нормальный вывод unit-тестов

```
33 passed, 2 warnings in 6.80s
```

- **`passed`** — тест прошёл, инвариант выполнен
- **`2 warnings`** — это `DeprecationWarning` от swig-биндингов `llama-cpp-python`, не наш код, игнорировать
- **`6.80s`** — нормальное время, большую часть занимает импорт `sentence_transformers`

### Упавший тест

```
FAILED src/tests/unit_test.py::TestFormatClusterOutput::test_small_tail_merges_into_previous_cluster
AssertionError: assert 2 == 1
```

Это означает что алгоритм сглаживания изменил поведение — хвостовой кластер больше не сливается с предыдущим. Нужно либо исправить код, либо (если изменение намеренное) обновить тест.

