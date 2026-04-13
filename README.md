# Conspectius Engine

[Русская версия](README.ru.md)

Research MVP for generating structured academic notes from long lecture audio.

The repository focuses on a simple idea: long, noisy lecture transcripts are easier to process as a multi-stage local pipeline than as a single giant prompt. Instead of asking one model to do everything at once, the project splits the task into transcription, cleaning, clustering, planning, and synthesis.

## Current State

This is a working research MVP, not a polished product.

What already works:

- `STT` with `faster-whisper`
- transcript cleaning with `Drafter`
- local semantic clustering
- local and global planning
- topic-to-cluster matching
- final conspect generation with `Synthesizer`

What is still experimental:

- `SmartCompressor`
- hallucination validation
- visualization/export pipeline
- stronger evaluation and automated tests

## Current Pipeline

Main `all` flow:

`audio -> STT -> Drafter -> Local Clustering -> Local Planner -> Global Planner -> Global Clustering -> Synthesizer`

The current version is optimized for:

- Russian-language lecture audio
- STEM / technical subjects
- local inference with limited VRAM
- inspectable intermediate artifacts saved to disk

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
  core/        # base abstractions, pipeline, STT, clustering, compression, utils
  configs/     # model configs, prompts, bad words
  tests/       # placeholders for smoke / unit / e2e tests

data/
  example-audio/
  example-transcrib/
  example-mini-conspect/
  example-clusters/
  example-plan/
  example-conspect/
```

## Outputs

The pipeline stores intermediate artifacts on disk so each step can be inspected separately:

- raw transcripts
- cleaned transcripts
- local clusters
- local plans
- global plans
- global clusters
- final generated conspects

This is intentional and useful for debugging, iteration, and future experiments.

## Demo

Placeholder for:

- pipeline diagram
- sample output screenshots
- before / after cleaning examples
- demo gif or short video

<!-- TODO: add demo media -->

## Example Results

Placeholder for:

- short lecture sample
- transcript fragment
- cleaned transcript fragment
- generated plan
- final conspect excerpt

## Tests

Test section placeholder.

Planned coverage:

- smoke tests
- unit tests for utilities
- integration tests for pipeline stages
- short end-to-end sample run

## Roadmap

- stabilize the current end-to-end path
- finish `SmartCompressor`
- improve math-aware post-processing
- add hallucination checks
- add export to `.md` / `.pdf` / `.docx`
- add benchmarks and evaluation
- add real automated tests

## Limitations

- focused mainly on Russian lecture audio
- best suited for technical / scientific content
- still sensitive to transcript noise
- not packaged as a user-facing application yet
- some parts are intentionally research stubs

## Citation

Placeholder for future paper / preprint.

```bibtex
@misc{conspectius_engine,
  title  = {Conspectius Engine},
  author = {TODO},
  year   = {2026},
  note   = {Work in progress}
}
```

## License

See [LICENSE](LICENSE).
