# LongConspectWriter

[Russian version](README.ru.md)

LongConspectWriter is a research prototype for turning long lecture audio into structured academic notes. The repository implements a local, multi-stage pipeline that separates transcription, transcript cleaning, semantic clustering, hierarchical planning, topic-aware synthesis, and Markdown export.

The default stack is local-first:

- STT: `faster-whisper`
- LLM stages: `llama_cpp` with GGUF weights
- semantic grouping: sentence embeddings plus clustering
- export: JSON-to-Markdown conversion for the final conspect

## What It Does

- transcribes long-form audio with VAD-aware STT
- removes transcript noise and keeps domain-relevant content
- groups sentences into local semantic clusters
- builds a two-level outline with local and global planning
- aligns clusters to chapter-level topics
- synthesizes a final structured JSON conspect
- exports the final result as Markdown for review and sharing

## System Overview

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer -> Markdown export`

The pipeline is intentionally staged. Each intermediate artifact is written to disk, which makes the system suitable for inspection, ablation studies, and later evaluation in a paper setting.

## Default Configuration

The bundled configs currently use:

- STT model: `large-v3-turbo`
- default GGUF model: `.models/T-lite-it-2.1-Q6_K.gguf`
- local clustering embedding model: `cointegrated/rubert-tiny2`
- global clustering embedding model: `intfloat/multilingual-e5-small`

By default, the LLM stages run locally through `llama_cpp`. The CLI currently reads the bundled YAML configs under `src/configs/config-agents/`.

## Quick Start

### Requirements

- Python `3.12+`
- `uv`
- a CUDA GPU is recommended for practical inference speed
- local access to the model files referenced in the configs

### Install

```bash
uv sync
```

### Run the full pipeline

```bash
uv run python __main__.py --action all --path_to_file "data/example-audio/your_lecture.mp3"
```

### Run individual stages

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

`all` runs the complete chain and then exports the final JSON conspect into Markdown.

## CLI Actions

| Action | Expected input | Result |
| --- | --- | --- |
| `all` | Audio file | Full pipeline with Markdown export |
| `stt` | Audio or video file | Raw transcript |
| `drafter` | Raw transcript text | Cleaned transcript |
| `local_clustering` | Transcript text | Sentence-level local clusters |
| `local_planner` | Local clusters text | Micro-topics |
| `global_planner` | Local plan text | JSON chapter outline |
| `planner` | Local clusters text | Local planning + global planning |
| `clustering` | Transcript text | Local clustering + planning + global clustering |
| `global_clustering` | Global plan JSON + local clusters text | Chapter-aligned clusters |
| `synthesizer` | Global clusters JSON | Final conspect JSON |

## Output Layout

The default configs write timestamped artifacts to `data/`:

- raw transcripts: `data/example-transcrib/`
- cleaned transcripts and synthesis JSON drafts: `data/example-conspect/`
- local clusters: `data/example-clusters/example-local-clusters/`
- local plans: `data/example-plan/example-local-plan/`
- global plans: `data/example-plan/example-global-plan/`
- global clusters: `data/example-clusters/example-global-clusters/`
- Markdown exports: `data/example-final-conspect/`

## Configuration

The main configuration files are:

- `src/configs/config-agents/stt/config_stt.yaml`
- `src/configs/config-agents/drafter/config_drafter.yaml`
- `src/configs/config-agents/local_planner/config_local_planner.yaml`
- `src/configs/config-agents/global_planner/config_global_planner.yaml`
- `src/configs/config-agents/synthesizer/config_synthesizer.yaml`
- `src/configs/config-agents/*/prompt_*.yaml`
- `src/configs/bad_words.py`
- `src/configs/ai_configs.py`

These files control:

- model selection
- generation parameters
- prompt templates
- output directories
- STT behavior
- generation cleanup rules

The current CLI path uses the bundled YAML files. A custom `--config_path` flow is reserved for future work.

## Repository Structure

```text
src/
  agents/      # drafter, planners, synthesizer
  core/        # STT, clustering, pipeline, utilities
  configs/     # dataclasses, prompts, bad words, YAML configs
  tests/       # test-oriented config fixtures

data/
  example-audio/
  example-transcrib/
  example-conspect/
  example-plan/
  example-clusters/
  example-final-conspect/

.models/
  # local GGUF weights for llama_cpp
```

## Research Notes

This repository is intentionally a research baseline rather than a polished end-user product. The staged design is useful for:

- qualitative inspection of intermediate outputs
- stage-wise ablations
- future benchmark creation
- paper writing and reproducibility

Current limitations:

- tuned primarily for Russian STEM lectures
- quality remains sensitive to transcript accuracy
- automatic evaluation is still limited
- the Markdown export is a lightweight formatter over the final JSON conspect

## Citation

```bibtex
@misc{longconspectwriter,
  title  = {LongConspectWriter},
  author = {TODO},
  year   = {2026},
  note   = {Research prototype; replace with final arXiv metadata}
}
```

## License

This repository is released under the [MIT License](LICENSE).
