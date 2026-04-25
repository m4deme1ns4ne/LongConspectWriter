# LongConspectWriter: Local Multi-Agent System For Generating Long Academic Conspect

[README.md in english](https://github.com/m4deme1ns4ne/LongConspectWriter/#longconspectwriter-local-multi-agent-system-for-generating-long-academic-conspect) | README.md на русском

## Оглавление
- [System Architecture](#system-architecture)
- [LongConspectWriter Deployment](#longconspectwriter-deployment)
- [CLI Actions](#cli-actions)
- [Output Artifacts](#output-artifacts)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Cases](#cases)

## System Architecture

```mermaid
flowchart TD
    Audio["Audio"] --> STT["STT<br/>(Generating transcription)"]

    STT --> LCluster["Local Clustering<br/>(Building local semantic chunks)"]
    LCluster --> LPlanner["Agent: Local Planner<br/>(Generating local topics)"]
    LPlanner --> GPlanner["Agent: Global Planner<br/>(Generating chapter outline)"]

    LCluster & GPlanner --> GCluster["Global Clustering<br/>(Aligning chunks to chapters)"]
    GCluster --> Synthesizer["Agent: Synthesizer<br/>(Rendering the academic summary)"]

    Synthesizer --> JSON["JSON conspect"]
    JSON --> MD["Markdown conspect"]
```

## LongConspectWriter Deployment

### Dependencies

- Python `3.12+`
- `uv`
- CUDA-совместимая среда
- локальные GGUF-веса

**Локальные GGUF-веса нужно скачивать отдельно, и сохранять в папку .models и в конфигах по пути src\configs\config-agents\ в файлах config_agentname.yaml указать для какого агента путь до весов.**

### Run the full pipeline

```bash
uv run python __main__.py --action all --path_to_file "data/example-audio/your_lecture.mp3"
```

`all` запускает полный пайплайн.

### Run individual stages pipeline

```bash
uv run python __main__.py --action stt --path_to_file "data/example-audio/your_lecture.mp3"
uv run python __main__.py --action local_clustering --path_to_file "data/example-transcrib/your_transcript.txt"
uv run python __main__.py --action local_planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action global_planner --path_to_file "data/example-plan/example-local-plan/your_local_plan.txt"
uv run python __main__.py --action planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action clustering --path_to_file "data/example-transcrib/your_transcript.txt"
uv run python __main__.py --action global_clustering --global_plan_path "data/example-plan/example-global-plan/your_global_plan.json" --local_clusters_path "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action synthesizer --path_to_file "data/example-clusters/example-global-clusters/your_global_clusters.json"
```

## CLI Actions

Каждый компонент пайплайна можно запускать отдельно для тестирования.

Таблица со всеми доступными командами:

| Action | Input | Output |
| --- | --- | --- |
| `all` | Аудио | Конспект в формате .md |
| `stt` | Аудио | Сырая транскрибация |
| `local_clustering` | Качественная транскрибация | Локальные кластеры |
| `local_planner` | Локальные кластеры | Локальные темы |
| `global_planner` | Локальные темы | Глобальные темы |
| `planner` | Локальные кластеры | Глобальные темы |
| `global_clustering` | Глобальные темы + локальные кластеры | Кластеры, привязанные к главам |
| `synthesizer` | Глобальные кластеры |  JSON-конспект |
| `clustering` | Качественная транскрибация | Глобальные темы |

## Output Artifacts

Промежуточные артефакты LongConspectWriter создаёт автоматически по ходу выполнения пайплайна.  
Они сохраняются в папке текущей сессии и раскладываются по стадиям:

- `01_stt/` - сырая транскрибация после FasterWhisper
- `02_local_clusters/` - локальные кластеры
- `03_local_planners/` - локальные темы
- `04_global_planners/` - план глав
- `05_global_clusters/` - глобальные кластеры
- `06_synthesizer/` - JSON-конспект
- `07_conspect_md/` - финальный Markdown-конспект

## Configuration

Основные конфиги расположены в `src/configs/config-agents/`:

Текущая конфигурация по умолчанию:

| Component | Default |
| --- | --- |
| STT | `large-v3-turbo` |
| LLM model | `.models/T-lite-it-2.1-Q5_K_M.gguf` |
| Local embeddings | `cointegrated/rubert-tiny2` |
| Global embeddings | `intfloat/multilingual-e5-small` |

Дополнительные dataclass-описания конфигураций находятся в `src/configs/configs.py`.

## Evaluation
...

## Cases

Примеры конспектов сгенерированных с помощью LongConspectWriter вы можете прочитать в папке [examples](examples).

Актуальные примеры находятся в папке: [examples/v1.5](examples/v1.5)
