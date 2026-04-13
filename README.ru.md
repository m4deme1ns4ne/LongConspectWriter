# Conspectius Engine

[English version](README.md)

Исследовательский MVP для генерации структурированного академического конспекта из длинной аудиозаписи лекции.

Главная идея проекта простая: длинную и шумную лекционную транскрипцию удобнее обрабатывать как многоэтапный локальный пайплайн, чем как один огромный промпт. Вместо того чтобы заставлять одну модель делать всё сразу, проект делит задачу на транскрибацию, очистку текста, кластеризацию, планирование и финальный синтез.

## Текущее состояние

Это рабочий исследовательский MVP, а не законченный продукт.

Что уже работает:

- `STT` на базе `faster-whisper`
- очистка транскрипции через `Drafter`
- локальная семантическая кластеризация
- локальный и глобальный planner
- привязка кластеров к темам
- генерация финального конспекта через `Synthesizer`

Что пока остаётся экспериментальным:

- `SmartCompressor`
- валидация галлюцинаций
- визуализация и экспорт
- полноценные автотесты и оценка качества

## Текущий пайплайн

Основной сценарий `all`:

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer`

Текущая версия оптимизирована под:

- русскоязычные лекции
- STEM / технические дисциплины
- локальный запуск при ограниченном VRAM
- сохранение промежуточных артефактов на диск

## Быстрый старт

### Требования

- Python `3.12+`
- желательно CUDA GPU
- желательно использовать `uv`

### Установка

```bash
uv sync
```

### Запуск полного пайплайна

```bash
uv run python __main__.py --action all --path_to_file "data/example-audio/your_lecture.mp3"
```

### Запуск отдельных этапов

```bash
uv run python __main__.py --action stt --path_to_file "data/example-audio/your_lecture.mp3"
uv run python __main__.py --action drafter --path_to_file "data/example-transcrib/your_transcript.txt"
uv run python __main__.py --action local_clustering --path_to_file "data/example-mini-conspect/your_cleaned_transcript.txt"
uv run python __main__.py --action planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action global_clustering --global_plan_path "data/example-plan/example-global-plan/plan.json" --local_clusters_path "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action synthesizer --path_to_file "data/example-clusters/example-global-clusters/global_clusters.json"
```

### Доступные CLI-действия

- `all`
- `stt`
- `drafter`
- `synthesizer`
- `planner`
- `local_planner`
- `global_planner`
- `clustering`
- `local_clustering`
- `global_clustering`

## Конфигурация

Основные конфиги лежат здесь:

- `src/configs/config.yaml`
- `src/configs/prompts.yaml`
- `src/configs/bad_words.py`

Через них можно менять:

- модели
- параметры генерации
- шаблоны промптов
- выходные директории
- настройки STT
- списки нежелательных фраз при генерации

## Структура проекта

```text
src/
  agents/      # Drafter, Planner, Synthesizer
  core/        # базовые абстракции, pipeline, STT, clustering, compression, utils
  configs/     # конфиги моделей, промпты, bad words
  tests/       # заготовки под smoke / unit / e2e тесты

data/
  example-audio/
  example-transcrib/
  example-mini-conspect/
  example-clusters/
  example-plan/
  example-conspect/
```

## Что сохраняется на диск

Пайплайн специально пишет промежуточные артефакты на диск, чтобы каждый шаг можно было отдельно посмотреть и отладить:

- сырые транскрипции
- очищенные транскрипции
- локальные кластеры
- локальные планы
- глобальные планы
- глобальные кластеры
- финальные конспекты

Это удобно для отладки, экспериментов и будущих абляций.

## Demo

Плейсхолдер под:

- схему пайплайна
- скриншоты результата
- примеры до / после очистки
- gif или короткое видео

<!-- TODO: add demo media -->

## Примеры результатов

Плейсхолдер под:

- короткий пример лекции
- фрагмент транскрипции
- фрагмент очищенного текста
- пример плана
- фрагмент финального конспекта

## Тесты

Пока это секция-заготовка.

Планируемое покрытие:

- smoke tests
- unit tests для утилит
- integration tests для этапов пайплайна
- короткий end-to-end прогон

## Дорожная карта

- стабилизировать текущий end-to-end сценарий
- довести `SmartCompressor`
- улучшить math-aware постобработку
- добавить проверки на галлюцинации
- добавить экспорт в `.md` / `.pdf` / `.docx`
- добавить бенчмарки и оценку качества
- добавить полноценные автотесты

## Ограничения

- проект в первую очередь заточен под русскоязычные лекции
- лучше всего подходит для технических и научных дисциплин
- всё ещё чувствителен к шумной транскрипции
- пока не упакован как пользовательское приложение
- часть модулей пока специально оставлена в виде research-заготовок

## Цитирование

Плейсхолдер под будущую статью / препринт.

```bibtex
@misc{conspectius_engine,
  title  = {LongConspectWriter},
  author = {TODO},
  year   = {2026},
  note   = {Work in progress}
}
```

## Лицензия

Смотри [LICENSE](LICENSE).
