# LongConspectWriter

[Русская версия](README.ru.md)

Research MVP for generating structured academic notes from long lecture audio.

The project uses a local multi-stage pipeline. Long, noisy lecture transcripts are processed step by step: transcription, transcript cleaning, semantic grouping, structure planning, topic matching, final long-form synthesis, and Markdown export of the last JSON result. The stack is hybrid: `faster-whisper` is used for STT, Transformers are used for Drafter and the planners, while Synthesizer can run either through `llama_cpp` on a local GGUF model or through a Transformers compatibility backend.

## What Works Now

- `STT` with `faster-whisper`
- transcript cleaning with `Drafter`
- local semantic clustering
- local and global planning
- global topic assignment
- final JSON conspect generation with `Synthesizer` on a local GGUF model or in Transformers compatibility mode
- Markdown export of the final JSON draft

This is a working research MVP, not a finished product.

## Current Pipeline

Main `all` flow:

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer -> Markdown export`

In the current version:

- `Drafter` cleans raw transcript noise before downstream processing and keeps only fragments with academic content
- `Synthesizer` works topic by topic, can split large topic clusters into smaller chunks, and produces the final structured JSON draft through the selected backend
- the final JSON draft is converted into a Markdown conspect at the end of the pipeline
- intermediate artifacts are saved to disk for inspection and debugging

## Scope

The project is currently aimed at:

- Russian-language lecture audio
- STEM / technical subjects
- local inference with limited VRAM
- long-form academic note generation instead of short summarization
- a mixed backend setup with Transformers and a local GGUF-based synthesizer

## Quick Start

### Requirements

- Python `3.12+`
- CUDA GPU recommended
- `uv` recommended

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
uv run python __main__.py --action local_clustering --path_to_file "data/example-mini-conspect/your_cleaned_transcript.txt"
uv run python __main__.py --action planner --path_to_file "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action global_clustering --global_plan_path "data/example-plan/example-global-plan/plan.json" --local_clusters_path "data/example-clusters/example-local-clusters/your_clusters.txt"
uv run python __main__.py --action synthesizer --path_to_file "data/example-clusters/example-global-clusters/global_clusters.json"
```

`all` now runs the complete chain and then exports the final JSON into a Markdown file.

### Available CLI actions

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

## Configuration

Main config files:

- `src/configs/config-agents/stt/config_stt.yaml`
- `src/configs/config-agents/drafter/config_drafter.yaml`
- `src/configs/config-agents/local_planner/config_local_planner.yaml`
- `src/configs/config-agents/global_planner/config_global_planner.yaml`
- `src/configs/config-agents/synthesizer/config_synthesizer.yaml`
- `src/configs/config-agents/*/prompt_*.yaml`
- `src/configs/bad_words.py`
- `src/configs/ai_configs.py`

You can change:

- model choices
- generation parameters
- prompt templates
- output directories
- STT options
- blocked phrases for generation cleanup
- `Synthesizer` backend selection and model path in `src/configs/config-agents/synthesizer/config_synthesizer.yaml`

## Project Structure

```text
src/
  agents/      # Drafter, Planner, Synthesizer
  core/        # base abstractions, pipeline, STT, clustering, utils
  configs/     # ai_configs, bad_words, config-agents
  tests/       # test placeholders and test configs

data/
  example-audio/
  example-transcrib/
  example-mini-conspect/
  example-clusters/
  example-plan/
  example-conspect/

.models/
  # local GGUF models for llama_cpp
```

## Saved Artifacts

The pipeline writes intermediate results to disk so each stage can be inspected separately:

- raw transcripts
- cleaned transcripts
- local clusters
- local plans
- global plans
- global clusters
- final generated JSON drafts
- exported Markdown conspects

This is intentional and useful for debugging, research iteration, and future evaluation.

## Tests

There is no full automated test suite yet, but the repository already contains test-oriented config files:

- `src/tests/test_config.yaml`
- `src/tests/test_prompts.yaml`

They can be used as lightweight fixtures for short local runs.

## Demo / Results

Placeholder for:

- pipeline diagram
- sample output screenshots
- before / after cleaning examples
- small end-to-end examples

## Limitations

- focused mainly on Russian lecture audio
- best suited for technical / scientific content
- still sensitive to transcript quality
- not packaged as a user-facing application yet
- evaluation is still manual / research-oriented
- the Synthesizer defaults to a local GGUF backend, with a Transformers fallback for compatibility
- final Markdown export is a simple formatter over the generated JSON, so malformed upstream output can still affect the result

## Citation

Placeholder for future paper / preprint.

```bibtex
@misc{long_conspect_writer,
  title  = {LongConspectWriter},
  author = {TODO},
  year   = {2026},
  note   = {Work in progress}
}
```

## License

This repository is currently licensed under the [MIT License](LICENSE).
