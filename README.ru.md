# LongConspectWriter

[English version](README.md)

Исследовательский MVP для генерации структурированного академического конспекта из длинной аудиозаписи лекции.

Проект использует локальный многоэтапный пайплайн вместо одного огромного промпта. Длинная и шумная лекционная транскрипция обрабатывается по шагам: транскрибация, очистка текста, семантическая группировка, планирование структуры, привязка тем и финальный длинный синтез.

## Что уже работает

- `STT` на базе `faster-whisper`
- очистка транскрипции через `Drafter`
- локальная семантическая кластеризация
- локальный и глобальный planner
- привязка кластеров к глобальным темам
- генерация финального конспекта через `Synthesizer`

Это рабочий исследовательский MVP, а не законченный продукт.

## Текущий пайплайн

Основной сценарий `all`:

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer`

В текущей версии:

- `Drafter` сначала очищает сырую транскрипцию от шума
- `Synthesizer` работает по темам и при необходимости режет большие кластеры на более мелкие куски
- промежуточные артефакты сохраняются на диск для просмотра и отладки

## На что сейчас ориентирован проект

- русскоязычные лекции
- STEM / технические дисциплины
- локальный запуск при ограниченном VRAM
- генерация длинного академического конспекта, а не короткой summary

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
- списки нежелательных фраз для очистки генерации

## Структура проекта

```text
src/
  agents/      # Drafter, Planner, Synthesizer
  core/        # базовые абстракции, pipeline, STT, clustering, utils
  configs/     # конфиги моделей, промпты, bad words
  tests/       # Tесты и тестовые конфиги

data/
  example-audio/
  example-transcrib/
  example-mini-conspect/
  example-clusters/
  example-plan/
  example-conspect/
```

## Какие артефакты сохраняются

Пайплайн специально пишет промежуточные результаты на диск, чтобы каждый шаг можно было отдельно проверить:

- сырые транскрипции
- очищенные транскрипции
- локальные кластеры
- локальные планы
- глобальные планы
- глобальные кластеры
- финальные конспекты

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

Сейчас проект использует [MIT License](LICENSE).
