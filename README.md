<p align="center">
  <img src="assets/banner.svg" alt="LongConspectWriter" width="100%">
</p>

# **LongConspectWriter: Overcoming Context Window Constraints in Audio Lecture Summarization with Local SLMs**

<p align="center">
  <b>English</b> •
  <a href="README.ru.md">Русский</a>
</p>

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/Habr-Read_the_article-65A3BE?style=for-the-badge&logo=habr&logoColor=white" alt="Article on Habr"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-A6CE39?style=for-the-badge" alt="License: MIT"></a>
</p>

LongConspectWriter turns a lecture audio/video recording into a structured academic PDF conspect with formulas and graphs — fully locally, on 8 GB of VRAM. No single LLM call ever receives the full transcript. Sample outputs are in [`examples/`](examples/); the method and evaluation are described in the [article](docs/article.md).

## Table of contents

- [System architecture](#system-architecture)
- [Installation and launch](#installation-and-launch)
- [CLI actions](#cli-actions)
- [Output artifacts](#output-artifacts)
- [Configuration](#configuration)
- [Evaluation](#evaluation)

## System architecture

<p align="center">
<img src="assets/mermaid-diagram-2026-06-05T23-14-56.svg" width="50%" alt="LongConspectWriter pipeline architecture">
</p>

### Core agents and components

| Component | Responsibility |
| --- | --- |
| `FasterWhisper` | Transcribes audio/video into text. |
| `SemanticLocalClusterizer` | Splits the transcript into local semantic clusters. |
| `AgentLocalPlanner` | Builds local topics from the clusters. |
| `AgentGlobalPlanner` | Assembles local topics into a global chapter plan. |
| `SemanticGlobalClusterizer` | Maps local clusters to the chapters of the global plan. |
| `AgentSynthesizerLlama` | Generates an academic JSON conspect and uses the extractor for context. |
| `_AgentExtractor` | Extracts entities from the current synthesis chunk to deduplicate the following chunks. |
| `convert_json_to_md` | Converts the synthesizer's JSON conspect into Markdown for the subsequent visualization stages. |
| `AgentGraphPlanner` | Analyzes the finished Markdown and inserts `[GRAPH_TYPE: ...]` placeholders next to quotes via normalized search. |
| `AgentGrapher` | Generates Python visualization code, runs it via `MPLBACKEND=Agg`, retries with an increasing temperature, and saves the graph mapping. |
| `add_graph_in_conspect` | Copies successful PNGs into the final `assets/` and replaces placeholders with HTML blocks containing the images. |
| `convert_md_to_pdf` | Converts the final Markdown into PDF via Playwright: renders HTML with MathJax for formulas and saves a paginated A4 document. |

## Installation and launch

### Dependencies

- Python `3.12+`
- `uv`

> The system was tested on a GeForce RTX 3050 8gb

### Running the full pipeline

```bash
uv run python __main__.py --action all --path_to_file "data/example-audio/your_lecture.mp3"
```

`all` runs the full scenario.

### Running individual stages

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

The optional `--lecture_theme` flag sets the lecture theme (`math`, `biology`, `chemistry`, etc.) and influences prompt selection in agents that support themed templates. If the flag is not passed, agents use the `universal` prompt.

Each CLI run creates a new session directory under `data/runs/<date>/<time>/`. If you run stages manually, pass the paths to the artifacts from the relevant session explicitly.

## CLI actions

Each pipeline component can be run separately for testing and debugging.

| Action | Input | Output |
| --- | --- | --- |
| `all` | Audio/video | Final PDF conspect with formulas and graphs |
| `stt` | Audio/video | `01_stt/out_filepath.json` with the raw transcription |
| `local_clustering` | STT transcript | `02_local_clusters/out_filepath.json` |
| `local_planner` | Local clusters | `03_local_planners/out_filepath.json` |
| `global_planner` | Local topics | `04_global_planners/out_filepath.json` |
| `planner` | Local clusters | Global plan via `local_planner -> global_planner` |
| `global_clustering` | Global plan + local clusters | `05_global_clusters/out_filepath.json` |
| `clustering` | STT transcript | Global clusters via `local_clustering -> planner -> global_clustering` |
| `synthesizer` | Global clusters | `06_synthesizer/conspect.json` |
| `convert_json_to_md` | JSON conspect | `07_conspect_md/conspect.md` |
| `graph_planner` | Markdown conspect | `08_graph_planner/out_filepath.md` with added `[GRAPH_TYPE: ...]` and `08_graph_planner/out_filepath.jsonl` |
| `grapher` | Markdown with `[GRAPH_TYPE: ...]` | `09_grapher/graphs_mapping.json`, `09_grapher/scripts/*.py`, `09_grapher/assets/*.png` |
| `add_graph_in_conspect` | Markdown with `[GRAPH_TYPE: ...]` + `graphs_mapping.json` | `10_conspect_with_graph_md/final_conspect.md` |
| `convert_md_to_pdf` | Markdown conspect | `11_conspect_pdf/final_conspect.pdf` |

## Output artifacts

Intermediate artifacts are created automatically in the current session folder:

```text
data/runs/YYYY.MM.DD/HH.MM.SS/
```

Main stage directories:

- `01_stt/` — raw transcription after FasterWhisper.
- `02_local_clusters/` — local semantic clusters.
- `03_local_planners/` — local topics.
- `04_global_planners/` — global chapter plan.
- `05_global_clusters/` — clusters mapped to global chapters.
- `05.1_extractor/` — JSONL output of the internal extractor during synthesis.
- `06_synthesizer/` — JSON conspect.
- `07_conspect_md/` — Markdown conspect without the final graph substitution.
- `08_graph_planner/` — Markdown after inserting `[GRAPH_TYPE: ...]` and the graph planner's per-chunk JSONL responses.
- `09_grapher/` — `graphs_mapping.json` and the generated graphs.
- `09_grapher/assets/` — PNG graphs created by `AgentGrapher`.
- `09_grapher/scripts/` — Python scripts used to render the graphs.
- `10_conspect_with_graph_md/` — final Markdown conspect.
- `10_conspect_with_graph_md/assets/` — local images copied for the final Markdown.
- `11_conspect_pdf/` — PDF version of the conspect with rendered formulas and graphs.

## Configuration

All configs are organized into three groups:

```
src/configs/
├── config_pipeline.yaml          — global pipeline parameters
├── config-agents/                — one config per agent
│   ├── stt/
│   ├── local_planner/
│   ├── global_planner/
│   ├── synthesizer/
│   ├── extractor/
│   ├── graph_planner/
│   └── grapher/
└── config-clusterizer/           — clustering parameters
    ├── config_local_clusterizer.yaml
    └── config_global_clusterizer.yaml
```

Dataclass descriptions of all config fields are in `src/configs/configs.py`.

### Environment variables

The `.env` file (template — `.env.example`) contains the HuggingFace token for downloading models:

```
HF_TOKEN=hf_...
```

Models are downloaded automatically into `.models/` on the first run.

## Evaluation

Quality was assessed with the LLM-as-a-judge method (judge — `Qwen3 Max Preview`) on 10 lectures from 5 domains, compared against the `Gemini 3.1 Pro` (single-call) baseline. Averaged across 7 paradigms, LCW scores **6.87/10 versus 8.84/10** for Gemini — ≈78% of the cloud reference quality, at zero inference cost and fully locally.

![Summary quality scores of LCW and Gemini 3.1 Pro conspects](assets/eval_comparison.png)

The methodology, paradigm-by-paradigm breakdown, and interpretation are in the [article](docs/article.md). The judge prompt, baseline prompt, dataset description, and full results are in the [evaluation/](evaluation/) folder ([judge prompt](evaluation/comparison/prompt_llm-as-a-judge.md), [dataset](evaluation/dataset.md)).
