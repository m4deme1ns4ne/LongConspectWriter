# LongConspectWriter

[Русская версия](README.ru.md)

Research MVP for generating structured academic notes from long lecture audio.

The project uses a local multi-stage pipeline instead of a single giant prompt. Long, noisy lecture transcripts are processed step by step: transcription, transcript cleaning, semantic grouping, structure planning, topic matching, and final long-form synthesis.

## What Works Now

- `STT` with `faster-whisper`
- transcript cleaning with `Drafter`
- local semantic clustering
- local and global planning
- global topic assignment
- final conspect generation with `Synthesizer`

This is a working research MVP, not a finished product.

## Current Pipeline

Main `all` flow:

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer`

In the current version:

- `Drafter` cleans raw transcript noise before downstream processing
- `Synthesizer` works topic by topic and can split large topic clusters into smaller chunks
- intermediate artifacts are saved to disk for inspection and debugging

## Scope

The project is currently aimed at:

- Russian-language lecture audio
- STEM / technical subjects
- local inference with limited VRAM
- long-form academic note generation instead of short summarization

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

- `src/configs/config.yaml`
- `src/configs/prompts.yaml`
- `src/configs/bad_words.py`

You can change:

- model choices
- generation parameters
- prompt templates
- output directories
- STT options
- blocked phrases for generation cleanup

## Project Structure

```text
src/
  agents/      # Drafter, Planner, Synthesizer
  core/        # base abstractions, pipeline, STT, clustering, utils
  configs/     # model configs, prompts, bad words
  tests/       # test placeholders and test configs

data/
  example-audio/
  example-transcrib/
  example-mini-conspect/
  example-clusters/
  example-plan/
  example-conspect/
```

## Saved Artifacts

The pipeline writes intermediate results to disk so each stage can be inspected separately:

- raw transcripts
- cleaned transcripts
- local clusters
- local plans
- global plans
- global clusters
- final generated conspects

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
