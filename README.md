# LongConspectWriter: Преодоление ограничений контекстного окна при конспектировании аудиолекций с помощью локальных малых языковых моделей

LongConspectWriter — локальная мультиагентная система для автоматического создания академических конспектов из аудио STEM-лекции. Система работает полностью офлайн на потребительском GPU.

## Оглавление

- [Архитектура системы](#архитектура-системы)
- [Требования](#требования)
- [Установка и запуск](#установка-и-запуск)
- [CLI-действия](#cli-действия)
- [Выходные артефакты](#выходные-артефакты)
- [Конфигурация](#конфигурация)
- [Evaluation](#evaluation)

## Архитектура системы

LongConspectWriter превращает аудио лекции в PDF-конспект. Пайплайн транскрибирует запись через FasterWhisper, строит локальные семантические кластеры, собирает глобальный план лекции, привязывает локальные кластеры к глобальным темам и синтезирует академический JSON-конспект. Во время синтеза внутренний `AgentExtractor` обновляет контекст лекции, чтобы следующие чанки не дублировали уже извлечённые сущности и темы.

После синтеза JSON конвертируется в Markdown. Отдельный `AgentGraphPlanner` анализирует готовый Markdown и вставляет `[GRAPH_TYPE: ...]`-плейсхолдеры там, где визуализация полезна и может быть сгенерирована кодом. Затем `AgentGrapher` находит эти плейсхолдеры, генерирует Python-скрипты для визуализаций, рендерит изображения с ретраями при ошибках и сохраняет mapping графиков. Этап `add_graph_in_conspect` заменяет плейсхолдеры HTML-блоками с локальными изображениями из `assets/`. Финальный этап `convert_md_to_pdf` конвертирует Markdown с графиками и LaTeX-формулами в PDF через Playwright + MathJax.

```mermaid
flowchart TD
    %% Входные данные
    Audio["Аудио / Видео"] --> STT["STT<br/>(Перевод аудио в текст)"]

    %% Кластеризация и планирование
    STT --> LCluster["Локальная кластеризация"]
    LCluster --> LPlanner["AgentLocalPlanner<br/>(Генерация локальных тем на каждый кластер)"]
    LPlanner --> GPlanner["AgentGlobalPlanner<br/>(Генерация глобальных тем на весь конспект)"]
    LCluster --> GCluster["Глобальная кластеризация<br/>(Привязка локальных кластеров к глобальным темам)"]
    GPlanner --> GCluster

    %% Синтез текста
    GCluster --> Synthesizer["AgentSynthesizer<br/>(Генерация текста конспекта)"]
    Synthesizer <--> Extractor["AgentExtractor<br/>(Обновление контекста/памяти)"]
    Synthesizer --> DraftJSON["Черновик конспекта"]

    %% Мультиагентный визуализатор
    DraftJSON --> GraphPlanner["AgentGraphPlanner<br/>(Генерация плейсхолдеров для графиков)"]
    GraphPlanner --> TaggedMD["Размеченный конспект с плейсхолдерами"]
    TaggedMD --> Grapher["AgentGrapher<br/>(Генерация кода для графиков)"]
    Grapher --> Images["Сгенерированные изображения<br/>(PNG)"]

    %% Финальная сборка
    TaggedMD --> FinalMD["Сборщик конспекта<br/>(Вставка HTML-тегов)"]
    Images -.-> FinalMD
    FinalMD["Конспект Markdown с графиками"]
    FinalMD --> FinalPDF["Конспект PDF с графиками"]
```

### Основные агенты и компоненты

| Component | Responsibility |
| --- | --- |
| `FasterWhisper` | Транскрибирует аудио/видео в текст в отдельном процессе. |
| `SemanticLocalClusterizer` | Делит транскрипт на локальные семантические кластеры. |
| `AgentLocalPlanner` | Строит локальные темы по кластерам. |
| `AgentGlobalPlanner` | Собирает локальные темы в глобальный план глав. |
| `SemanticGlobalClusterizer` | Привязывает локальные кластеры к главам глобального плана. |
| `AgentSynthesizerLlama` | Генерирует академический JSON-конспект и использует extractor для контекста. |
| `AgentExtractor` | Извлекает сущности из текущего чанка синтеза для дедупликации следующих чанков. |
| `AgentGraphPlanner` | Анализирует готовый Markdown и вставляет `[GRAPH_TYPE: ...]`-плейсхолдеры по цитатам через нормализованный поиск. |
| `AgentGrapher` | Генерирует Python-код визуализации, запускает его через `MPLBACKEND=Agg`, делает ретраи с повышением температуры и сохраняет mapping графиков. |
| `add_graph_in_conspect` | Копирует успешные PNG в финальные `assets/` и заменяет плейсхолдеры HTML-блоками с изображениями. |
| `convert_md_to_pdf` | Конвертирует финальный Markdown в PDF через Playwright: рендерит HTML с MathJax для формул и сохраняет постраничный A4-документ. |

## Установка и запуск

### Зависимости

- Python `3.12+`
- `uv`

> Система тестировалась на GeForce RTX 3050 8gb

### Запуск полного пайплайна

```bash
uv run python __main__.py --action all --path_to_file "data/example-audio/your_lecture.mp3"
```

`all` запускает полный сценарий:

```text
STT -> local clustering -> local planner -> global planner -> global clustering -> synthesizer -> JSON to Markdown -> graph planner -> grapher -> final Markdown with images -> PDF
```

### Запуск отдельных стадий

```bash
uv run python __main__.py --action stt --path_to_file "data/example-audio/your_lecture.mp3"
uv run python __main__.py --action local_clustering --path_to_file "data/example-transcrib/your_transcript.json"
uv run python __main__.py --action local_planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.json"
uv run python __main__.py --action global_planner --path_to_file "data/example-plan/example-local-plan/your_local_plan.json"
uv run python __main__.py --action planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.json"
uv run python __main__.py --action global_clustering --global_plan_path "data/example-plan/example-global-plan/your_global_plan.json" --local_clusters_path "data/example-clusters/example-local-clusters/your_clusters.json"
uv run python __main__.py --action clustering --path_to_file "data/example-transcrib/your_transcript.json"
uv run python __main__.py --action synthesizer --path_to_file "data/example-clusters/example-global-clusters/your_global_clusters.json"
uv run python __main__.py --action convert_json_to_md --path_to_file "data/runs/YYYY.MM.DD/HH.MM.SS/06_synthesizer/conspect.json"
uv run python __main__.py --action graph_planner --path_to_file "data/runs/YYYY.MM.DD/HH.MM.SS/07_conspect_md/conspect.md"
uv run python __main__.py --action grapher --path_to_file "data/runs/YYYY.MM.DD/HH.MM.SS/08_graph_planner/out_filepath.md"
uv run python __main__.py --action add_graph_in_conspect --path_to_file "data/runs/YYYY.MM.DD/HH.MM.SS/08_graph_planner/out_filepath.md" --graphs_path "data/runs/YYYY.MM.DD/HH.MM.SS/09_grapher/graphs_mapping.json"
uv run python __main__.py --action convert_md_to_pdf --path_to_file "data/runs/YYYY.MM.DD/HH.MM.SS/10_conspect_with_graph_md/final_conspect.md"
```

Необязательный флаг `--lecture_theme` задаёт тему лекции (`math`, `biology`, `chemistry` и т. д.) и влияет на выбор промпта в агентах, поддерживающих тематические шаблоны. Если флаг не передан, агенты используют `universal`-промпт.

Каждый запуск CLI создаёт новую сессионную директорию внутри `data/runs/<date>/<time>/`. Если вы запускаете стадии вручную, передавайте пути к артефактам из нужной сессии явно.

## CLI-действия

Каждый компонент пайплайна можно запускать отдельно для тестирования и отладки.

| Action | Input | Output |
| --- | --- | --- |
| `all` | Аудио/видео | Финальный PDF-конспект с формулами и графиками |
| `stt` | Аудио/видео | `01_stt/out_filepath.json` с сырой транскрибацией |
| `local_clustering` | Транскрипт STT | `02_local_clusters/out_filepath.json` |
| `local_planner` | Локальные кластеры | `03_local_planners/out_filepath.json` |
| `global_planner` | Локальные темы | `04_global_planners/out_filepath.json` |
| `planner` | Локальные кластеры | Глобальный план через `local_planner -> global_planner` |
| `global_clustering` | Глобальный план + локальные кластеры | `05_global_clusters/out_filepath.json` |
| `clustering` | Транскрипт STT | Глобальные кластеры через `local_clustering -> planner -> global_clustering` |
| `synthesizer` | Глобальные кластеры | `06_synthesizer/conspect.json` |
| `convert_json_to_md` | JSON-конспект | `07_conspect_md/conspect.md` |
| `graph_planner` | Markdown-конспект | `08_graph_planner/out_filepath.md` с добавленными `[GRAPH_TYPE: ...]` и `08_graph_planner/out_filepath.jsonl` |
| `grapher` | Markdown с `[GRAPH_TYPE: ...]` | `09_grapher/graphs_mapping.json`, `09_grapher/scripts/*.py`, `09_grapher/assets/*.png` |
| `add_graph_in_conspect` | Markdown с `[GRAPH_TYPE: ...]` + `graphs_mapping.json` | `10_conspect_with_graph_md/final_conspect.md` |
| `convert_md_to_pdf` | Markdown-конспект | `11_conspect_pdf/final_conspect.pdf` |

## Выходные артефакты

Промежуточные артефакты создаются автоматически в папке текущей сессии:

```text
data/runs/YYYY.MM.DD/HH.MM.SS/
```

Основные stage-директории:

- `01_stt/` — сырая транскрибация после FasterWhisper.
- `02_local_clusters/` — локальные семантические кластеры.
- `03_local_planners/` — локальные темы.
- `04_global_planners/` — глобальный план глав.
- `05_global_clusters/` — кластеры, привязанные к глобальным главам.
- `05.1_extractor/` — JSONL-вывод внутреннего extractor во время синтеза.
- `06_synthesizer/` — JSON-конспект.
- `07_conspect_md/` — Markdown-конспект без финальной подстановки графиков.
- `08_graph_planner/` — Markdown после вставки `[GRAPH_TYPE: ...]` и JSONL-ответы graph planner по чанкам.
- `09_grapher/` — `graphs_mapping.json` и сгенерированные графики.
- `09_grapher/assets/` — PNG-графики, созданные `AgentGrapher`.
- `09_grapher/scripts/` — Python-скрипты, которыми рендерились графики.
- `10_conspect_with_graph_md/` — финальный Markdown-конспект.
- `10_conspect_with_graph_md/assets/` — локальные изображения, скопированные для финального Markdown.
- `11_conspect_pdf/` — PDF-версия конспекта с отрендеренными формулами и графиками.

## Конфигурация

Главный конфиг пайплайна находится в `src/configs/config_pipeline.yaml`:

Конфиги агентов расположены в `src/configs/config-agents/`, конфиги кластеризации — в `src/configs/config-clusterizer/`.

Dataclass-описания конфигураций находятся в `src/configs/configs.py`.

## Evaluation

Оценка качества конспектов проводилась методом LLM-as-a-judge по 7 парадигмам (P1–P7). Промпт судьи: [llm-as-a-judge](evaluation/comparison/prompt_llm-as-a-judge.md).
Полная методология и датасет — в папке [evaluation/](evaluation/).

**Baseline** — Gemini 3.1 Pro с [детальным системным промптом](evaluation/comparison/prompt_gemini.md). 
В отличие от LCW, Gemini обрабатывает полный транскрипт целиком и работал через оффициальный сайт [Gemini](https://gemini.google.com) с подпиской Google AI Pro.

**Датасет:** 10 лекций из 5 предметных областей: алгоритмы, машинное обучение, мат. анализ, биология, химия.

### Сводные оценки

<p align="center">
<img src="assets/Снимок экрана 2026-05-14 235216.png" width="95%" alt="summary scores">
</p>

## Детализация по парадигмам. _Формат: LCW / Gemini_

<p align="center">
<img src="assets/Снимок экрана 2026-05-14 235127.png" width="95%" alt="paradigm scores by P1–P7">
</p>

**Вывод.** LCW достигает **78% качества SOTA модели** (6.87 / 10 против 8.84 / 10 у Gemini), работая полностью офлайн на потребительском GPU с 8B-моделями. 

**Сильные стороны.** Покрытие материала (P6: 8.8 / 9.3) и педагогическая глубина (P7: 8.1 / 9.1) — показатели, наиболее близкие к Gemini. Это говорит о том, что связка локальных планировщиков и глобального плана эффективно решает задачу «не упустить тему» даже без сквозного контекста. Структурность конспекта (P2: 8.6 / 9.9) также высока — агент-синтезатор стабильно выдаёт логичную иерархию разделов.

**Ограничения.** Фактическая точность (P5: 7.2 / 10.0) — показатель, наиболее чувствительный к размеру модели: 8B-модель чаще галлюцинирует детали и иногда переформулирует термины. LaTeX-формализм (P3: 5.7 / 8.6) страдает по той же причине: корректная разметка формул требует точного воспроизведения синтаксиса, с чем малые модели справляются хуже. P4 (визуализация) низок у обеих систем (LCW: 2.3, Gemini: 5.4) — автоматический выбор уместного типа графика и его корректная генерация объективно сложны вне зависимости от размера модели.

**По доменам** лучший результат показывают Алгоритмы (86%): контент хорошо структурирован в транскрипте и не требует сложной математической нотации. Наибольший разрыв — в Машинном обучении (72%) и Химии (73%): лекции плотно насыщены формульной нотацией и специализированной терминологией, где дефицит параметров ощущается острее всего.
