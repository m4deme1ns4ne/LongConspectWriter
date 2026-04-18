# LongConspectWriter

[English version](README.md)

LongConspectWriter - исследовательский прототип для преобразования длинных аудиозаписей лекций в структурированные академические конспекты. Репозиторий реализует локальный многоэтапный пайплайн, в котором транскрибация, очистка транскрипта, семантическая кластеризация, иерархическое планирование, тематический синтез и экспорт в Markdown разделены на независимые воспроизводимые стадии.

Стек по умолчанию локальный:

- STT: `faster-whisper`
- LLM-стадии: `llama_cpp` с весами GGUF
- семантическая группировка: эмбеддинги предложений и кластеризация
- экспорт: преобразование JSON в Markdown для финального конспекта

## Что делает проект

- транскрибирует длинные лекции с учетом VAD
- очищает транскрипт от шума и оставляет только предметно значимые фрагменты
- группирует предложения в локальные семантические кластеры
- строит двухуровневый план с локальным и глобальным планированием
- сопоставляет кластеры главам верхнего уровня
- синтезирует итоговый структурированный JSON-конспект
- экспортирует результат в Markdown для просмотра и распространения

## Схема пайплайна

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer -> Markdown export`

Пайплайн намеренно разбит на стадии. Каждый промежуточный артефакт сохраняется на диск, поэтому систему удобно использовать для ручной проверки, абляционных экспериментов и последующей оценки в формате статьи.

## Конфигурация по умолчанию

Встроенные конфиги сейчас используют:

- модель STT: `large-v3-turbo`
- модель GGUF по умолчанию: `.models/T-lite-it-2.1-Q6_K.gguf`
- модель эмбеддингов для локальной кластеризации: `cointegrated/rubert-tiny2`
- модель эмбеддингов для глобальной кластеризации: `intfloat/multilingual-e5-small`

По умолчанию LLM-стадии работают локально через `llama_cpp`. CLI сейчас читает только встроенные YAML-конфиги из `src/configs/config-agents/`.

## Быстрый старт

### Требования

- Python `3.12+`
- `uv`
- желательно наличие CUDA GPU для приемлемой скорости инференса
- локальный доступ к файлам моделей, указанным в конфигурации

### Установка

```bash
uv sync
```

### Запуск полного пайплайна

```bash
uv run python __main__.py --action all --path_to_file "data/example-audio/your_lecture.mp3"
```

### Запуск отдельных стадий

```bash
uv run python __main__.py --action stt --path_to_file "data/example-audio/your_lecture.mp3"
uv run python __main__.py --action drafter --path_to_file "data/example-transcrib/your_transcript.txt"
uv run python __main__.py --action local_clustering --path_to_file "data/example-transcrib/your_transcript.txt"
uv run python __main__.py --action local_planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action global_planner --path_to_file "data/example-plan/example-local-plan/your_local_plan.txt"
uv run python __main__.py --action planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action clustering --path_to_file "data/example-transcrib/your_transcript.txt"
uv run python __main__.py --action global_clustering --global_plan_path "data/example-plan/example-global-plan/your_global_plan.json" --local_clusters_path "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action synthesizer --path_to_file "data/example-clusters/example-global-clusters/your_global_clusters.json"
```

`all` запускает всю цепочку и затем экспортирует финальный JSON-конспект в Markdown.

## Действия CLI

| Действие | Ожидаемый вход | Результат |
| --- | --- | --- |
| `all` | Аудиофайл | Полный пайплайн с экспортом в Markdown |
| `stt` | Аудио- или видеофайл | Сырой транскрипт |
| `drafter` | Текст сырого транскрипта | Очищенный транскрипт |
| `local_clustering` | Текст транскрипта | Локальные кластеры на уровне предложений |
| `local_planner` | Текст локальных кластеров | Микротемы |
| `global_planner` | Текст локального плана | JSON-оглавление глав |
| `planner` | Текст локальных кластеров | Локальное и глобальное планирование |
| `clustering` | Текст транскрипта | Локальная кластеризация, планирование и глобальная кластеризация |
| `global_clustering` | JSON глобального плана + текст локальных кластеров | Кластеры, привязанные к главам |
| `synthesizer` | JSON глобальных кластеров | Итоговый JSON-конспект |

## Выходные артефакты

Конфиги по умолчанию сохраняют артефакты с временными метками в `data/`:

- сырые транскрипты: `data/example-transcrib/`
- очищенные транскрипты и JSON-черновики синтеза: `data/example-conspect/`
- локальные кластеры: `data/example-clusters/example-local-clusters/`
- локальные планы: `data/example-plan/example-local-plan/`
- глобальные планы: `data/example-plan/example-global-plan/`
- глобальные кластеры: `data/example-clusters/example-global-clusters/`
- Markdown-экспорт: `data/example-final-conspect/`

## Конфигурация

Основные конфигурационные файлы:

- `src/configs/config-agents/stt/config_stt.yaml`
- `src/configs/config-agents/drafter/config_drafter.yaml`
- `src/configs/config-agents/local_planner/config_local_planner.yaml`
- `src/configs/config-agents/global_planner/config_global_planner.yaml`
- `src/configs/config-agents/synthesizer/config_synthesizer.yaml`
- `src/configs/config-agents/*/prompt_*.yaml`
- `src/configs/bad_words.py`
- `src/configs/ai_configs.py`

Через них настраиваются:

- выбор моделей
- параметры генерации
- шаблоны промптов
- выходные директории
- поведение STT
- правила очистки генерации

Сейчас CLI использует встроенные YAML-конфиги. Ветка для пользовательского `--config_path` пока зарезервирована на будущее.

## Структура репозитория

```text
src/
  agents/      # drafter, planners, synthesizer
  core/        # STT, clustering, pipeline, utilities
  configs/     # dataclasses, prompts, bad words, YAML configs
  tests/       # тестовые конфигурационные фикстуры

data/
  example-audio/
  example-transcrib/
  example-conspect/
  example-plan/
  example-clusters/
  example-final-conspect/

.models/
  # локальные GGUF-веса для llama_cpp
```

## Исследовательские заметки

Этот репозиторий задуман как исследовательская база, а не как отполированный пользовательский продукт. Многостадийная архитектура удобна для:

- ручного анализа промежуточных результатов
- абляционных экспериментов по стадиям
- последующего построения бенчмарка
- подготовки статьи и воспроизводимых экспериментов

Текущие ограничения:

- проект в первую очередь настроен на русскоязычные STEM-лекции
- качество результата по-прежнему сильно зависит от точности транскрипции
- автоматическая оценка пока ограничена
- Markdown-экспорт является легким форматтером поверх итогового JSON-конспекта

## Цитирование

```bibtex
@misc{longconspectwriter,
  title  = {LongConspectWriter},
  author = {TODO},
  year   = {2026},
  note   = {Исследовательский прототип; замените на финальные метаданные arXiv}
}
```

## Лицензия

Проект распространяется под [MIT License](LICENSE).
