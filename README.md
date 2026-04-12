# Conspectius Engine

Research MVP for turning long lecture audio into structured academic notes.

The project is focused on a simple idea: instead of sending a noisy long transcript into one model and hoping for the best, break the task into several smaller stages that are easier to run locally and easier to improve independently.

At the moment, the repository already has a working end-to-end path from audio to generated notes. Some modules are still experimental and intentionally kept outside the main pipeline while the core flow is being tested.

<!-- TODO: add banner / teaser image -->
<!-- Example:
![Conspectius Demo](docs/images/teaser.png)
-->

## What It Does

- Transcribes lecture audio with `faster-whisper`
- Splits transcript into local semantic blocks
- Builds a lightweight chapter plan
- Matches local blocks to global topics
- Generates a structured long-form conspect with an LLM

Current target domain:

- Russian-language lectures
- STEM / technical subjects
- Long-form academic speech
- Local inference with limited VRAM

## Current Status

This repository is a **working research MVP**, not a polished product.

### In the main pipeline now

- `STT`
- `Local clustering`
- `Local planner`
- `Global planner`
- `Global clustering`
- `Synthesizer`

### Experimental / not yet in the final flow

- `Drafter`
- `SmartCompressor`
- specialist routing / domain normalization
- visual generation / export pipeline

This split is intentional: the stable path is used to validate the overall concept first, while the heavier modules are iterated separately.

## Pipeline

### Current working flow

`audio -> STT -> local clustering -> planning -> global clustering -> synthesizer`

### Intended full flow

`audio -> STT -> local clustering -> drafter / compression -> planning -> global clustering -> synthesizer -> export`

## Quick Start

### Requirements

- Python `3.12+`
- CUDA GPU recommended
- `uv` recommended for environment setup

### Install

```bash
uv sync
```

### Environment

Create `.env` if needed:

```env
# add your local settings here
```

### Run the full pipeline

```bash
uv run python __main__.py --action all --path_to_file "data/example-audio/your_lecture.mp3"
```

### Run individual stages

```bash
uv run python __main__.py --action stt --path_to_file "data/example-audio/your_lecture.mp3"
uv run python __main__.py --action local_clustering --path_to_file "data/example-transcrib/your_transcript.txt"
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

Main configs live in:

- `src/configs/config_agents.yaml`
- `src/configs/prompts.yaml`

From there you can change:

- model names
- generation parameters
- prompt templates
- output directories
- STT settings

## Project Structure

```text
src/
  agents/      # LLM agents: drafter, planner, synthesizer
  core/        # pipeline, STT, clustering, compression, VRAM helpers
  configs/     # YAML configs and prompts
  tests/       # placeholders for smoke / unit / e2e tests

data/
  example-audio/
  example-transcrib/
  example-clusters/
  example-plan/
  example-conspect/
  example-mini-conspect/
```

## Outputs

The pipeline currently writes intermediate artifacts to disk so each stage can be inspected separately:

- transcripts
- local clusters
- local plans
- global plans
- global clusters
- generated conspects

This makes debugging much easier and is useful for research iteration.

## Demo

Placeholder for:

- pipeline screenshot
- sample generated conspect
- before / after examples
- short demo gif

<!-- TODO: add demo media -->
<!--
![Pipeline](docs/images/pipeline.png)
![Sample Output](docs/images/sample-output.png)
-->

## Example Results

Placeholder for:

- one short input lecture example
- one transcript fragment
- one planning example
- one final conspect excerpt

This section is intentionally left open for future updates.

## Tests

Test section placeholder.

Planned coverage:

- [ ] smoke tests
- [ ] unit tests for core utilities
- [ ] integration tests for pipeline stages
- [ ] end-to-end run on a short sample

Suggested future commands:

```bash
uv run pytest
```

## Roadmap

- stabilize the current end-to-end path
- reconnect `Drafter` into the main flow
- implement `SmartCompressor`
- improve math-aware post-processing
- add export to `.md` / `.pdf` / `.docx`
- add reproducible benchmarks
- add proper automated tests

## Limitations

- optimized primarily for Russian lecture audio
- best suited for technical / scientific material
- still sensitive to transcript noise
- not yet packaged as a user-facing product
- some modules are research stubs by design

## Why This Repo Exists

The goal is not just "summarization".

The goal is a local, extensible pipeline for generating **structured academic conspects** from long, noisy lecture audio under real hardware limits.

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
