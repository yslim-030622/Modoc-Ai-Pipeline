# MoDoc AI Pipeline MVP

This repository contains a semi-automated MVP for converting MoDoc Q&A blog
content into short-form video planning artifacts. The first milestone is not a
fully automated video publisher. It is a repeatable pipeline that reads source
Q&A rows, generates multilingual scripts, creates a medical-review packet, and
records the human time needed for KPI tracking.

## What is automated now

- Read Q&A rows from `Q&A Blog Contents List.xlsx`.
- Prefer rows whose `Status (English)` is `Published`.
- Generate short-form scripts in English, Korean, and Spanish with Gemini.
- Generate medical claims and evidence references for reviewer support.
- Write a reviewer-friendly Markdown packet.
- Append a timing row to `logs/pipeline_runs.csv`.

## What is still manual

- Medical review pass/fail.
- Text-to-speech.
- Video rendering.
- YouTube, MoDoc blog, Instagram, and Facebook publishing.
- Final view-count collection.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

Create `.env` from the example and add your Gemini key:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Do not commit `.env`. It is ignored by git.

## Run the first generation

```bash
python3 -m modoc_pipeline generate --input "Q&A Blog Contents List.xlsx" --limit 1
```

You can target one Excel row directly:

```bash
python3 -m modoc_pipeline generate --input "Q&A Blog Contents List.xlsx" --row 3
```

You can also record human time during the run:

```bash
python3 -m modoc_pipeline generate \
  --input "Q&A Blog Contents List.xlsx" \
  --limit 1 \
  --content-selection-minutes 2 \
  --format-decision-minutes 1 \
  --script-generation-minutes 3 \
  --notes "First baseline run"
```

The timing values are human intervention minutes. Automated API wait time is
stored separately in `status.json` because the internship KPI denominator is
meant to measure repeated human work.

## Generated files

Each processed Q&A row creates a folder under `outputs/{run_id}/`:

- `source.json`: normalized source Q&A and metadata.
- `scripts.json`: English, Korean, and Spanish short-form scripts.
- `claims.json`: medical claims and evidence references.
- `review_packet.md`: reviewer-friendly source, scripts, claims, and notes.
- `status.json`: model, run status, prompt version, and generation timing.
- `raw_gemini_response.txt`: original model output for debugging.

The pipeline also appends one row to:

```text
logs/pipeline_runs.csv
```

That CSV is the starting point for KPI denominator tracking.

## Monday submission status format

Use this format when sharing progress:

```text
GitHub repo:
Current working features:
Sample output path:
Known limitations:
Next steps:
```

Suggested current status:

```text
GitHub repo: <repo link>
Current working features: Excel ingestion, Gemini multilingual script generation,
medical-review packet generation, KPI timing CSV logging.
Sample output path: outputs/<run_id>/review_packet.md
Known limitations: No TTS, video rendering, medical review workflow UI, or
platform upload automation yet.
Next steps: Add TTS + FFmpeg render, reviewer status tracking, and upload helpers.
```
