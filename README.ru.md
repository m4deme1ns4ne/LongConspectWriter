# LongConspectWriter

[English version](README.md)

Исследовательский MVP для генерации структурированного академического конспекта из длинной аудиозаписи лекции.

Проект использует локальный многоэтапный пайплайн. Длинная и шумная лекционная транскрипция обрабатывается по шагам: транскрибация, очистка текста, семантическая группировка, планирование структуры, привязка тем, финальный синтез и экспорт результата в Markdown. Стек гибридный: `faster-whisper` используется для `STT`, `Transformers` используются для `Drafter` и планировщиков, а `Synthesizer` может работать либо через `llama_cpp` на локальной `GGUF`-модели, либо через совместимый бэкенд `Transformers`.

## Что уже работает

- `STT` на базе `faster-whisper`
- очистка транскрипции через `Drafter` с отсечением фрагментов без предметного смысла
- локальная семантическая кластеризация
- локальный и глобальный planner
- привязка кластеров к глобальным темам
- генерация финального JSON-черновика через `Synthesizer` на локальной `GGUF`-модели или в режиме совместимости через `Transformers`
- преобразование финального JSON в Markdown-конспект

Это рабочий исследовательский MVP, а не законченный продукт.

## Текущий пайплайн

Основной сценарий `all`:

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer -> Markdown export`

В текущей версии:

- `Drafter` сначала очищает сырую транскрипцию от шума
- `Drafter` старается оставлять только фрагменты, где есть хотя бы одна полезная предметная мысль
- `Synthesizer` работает по темам и при необходимости режет большие кластеры на более мелкие куски
- `Synthesizer` пишет финальный JSON-черновик через выбранный бэкенд
- финальный JSON-черновик автоматически переводится в Markdown в конце пайплайна
- промежуточные артефакты сохраняются на диск для просмотра и отладки

## На что сейчас ориентирован проект

- русскоязычные лекции
- STEM / технические дисциплины
- локальный запуск при ограниченном VRAM
- генерация длинного академического конспекта, а не короткой summary
- гибридная схема, где часть агентов работает через `Transformers`, а `Synthesizer` выбирает между `llama_cpp` и `Transformers`

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

`all` теперь прогоняет всю цепочку и затем экспортирует финальный JSON в Markdown-файл.

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

- `src/configs/config-agents/stt/config_stt.yaml`
- `src/configs/config-agents/drafter/config_drafter.yaml`
- `src/configs/config-agents/local_planner/config_local_planner.yaml`
- `src/configs/config-agents/global_planner/config_global_planner.yaml`
- `src/configs/config-agents/synthesizer/config_synthesizer.yaml`
- `src/configs/config-agents/*/prompt_*.yaml`
- `src/configs/bad_words.py`
- `src/configs/ai_configs.py`

Через них можно менять:

- модели
- параметры генерации
- шаблоны промптов
- выходные директории
- настройки STT
- списки нежелательных фраз для очистки генерации
- выбор бэкенда `Synthesizer` и путь к модели в `src/configs/config-agents/synthesizer/config_synthesizer.yaml`

## Структура проекта

```text
src/
  agents/      # Drafter, Planner, Synthesizer
  core/        # базовые абстракции, pipeline, STT, clustering, utils
  configs/     # ai_configs, bad_words, config-agents
  tests/       # Tесты и тестовые конфиги

data/
  example-audio/
  example-transcrib/
  example-mini-conspect/
  example-clusters/
  example-plan/
  example-conspect/

.models/
  # локальные GGUF-модели для llama_cpp
```

## Какие артефакты сохраняются

Пайплайн специально пишет промежуточные результаты на диск, чтобы каждый шаг можно было отдельно проверить:

- сырые транскрипции
- очищенные транскрипции
- локальные кластеры
- локальные планы
- глобальные планы
- глобальные кластеры
- финальные JSON-черновики
- экспортированные Markdown-конспекты

## Тесты

Пока полноценного автотестового контура нет, но в репозитории уже есть тестовые конфиги:

- `src/tests/test_config.yaml`
- `src/tests/test_prompts.yaml`

Их можно использовать как лёгкие фикстуры для коротких локальных прогонов.

## Demo / Примеры результатов

Плейсхолдер под:

- схему пайплайна
- скриншоты результата
- примеры до / после очистки
- маленькие end-to-end примеры

## Ограничения

- проект в первую очередь заточен под русскоязычные лекции
- лучше всего подходит для технических и научных дисциплин
- всё ещё чувствителен к качеству транскрипции
- пока не упакован как пользовательское приложение
- оценка качества пока в основном ручная / исследовательская
- по умолчанию `Synthesizer` использует локальный `GGUF`-бэкенд, но есть и режим совместимости через `Transformers`
- финальный Markdown — это простое преобразование сгенерированного JSON, поэтому ошибки на предыдущих стадиях всё ещё могут влиять на результат

## Цитирование

Плейсхолдер под будущую статью / препринт.

```bibtex
@misc{long_conspect_writer,
  title  = {LongConspectWriter},
  author = {TODO},
  year   = {2026},
  note   = {Work in progress}
}
```

## Лицензия

Проект использует [MIT License](LICENSE).
