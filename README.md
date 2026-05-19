# MoDoc AI Pipeline MVP

This repository contains a semi-automated MVP for converting MoDoc Q&A blog
content into short-form video artifacts. The first milestone is a repeatable
pipeline that reads source Q&A rows, generates multilingual scripts, creates a
medical-review packet, plans video scenes, optionally generates Veo clips, and
records the human time needed for KPI tracking.

## What is automated now

- Read Q&A rows from `Q&A Blog Contents List.xlsx`.
- Prefer rows whose `Status (English)` is `Published`.
- Generate short-form scripts in English, Korean, and Spanish with Gemini.
- Generate medical claims and evidence references for reviewer support.
- Write a reviewer-friendly Markdown packet.
- Append a timing row to `logs/pipeline_runs.csv`.
- Plan 9:16 Veo scenes from generated scripts.
- Generate Vertex AI Veo clips when Google Cloud credentials are configured.
- Render final MP4 files with burned subtitles when FFmpeg is installed.

## What is still manual

- Medical review pass/fail.
- YouTube, MoDoc blog, Instagram, and Facebook publishing.
- Final view-count collection.
- Dedicated text-to-speech. Current video audio comes from Veo-generated audio.

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

VERTEX_PROJECT_ID=your-google-cloud-project-id
VERTEX_PROJECT_NAME=projects/your-google-cloud-project-number
VERTEX_LOCATION=us-central1
VEO_MODEL=veo-3.0-generate-001
VEO_PERSON_GENERATION=allow_all
VEO_GENERATE_AUDIO=true
```

Do not commit `.env`. It is ignored by git.

For Veo generation, install and authenticate Google Cloud SDK:

```bash
brew install --cask google-cloud-sdk
gcloud auth application-default login
gcloud config set project 47589570665
```

For final MP4 rendering, install FFmpeg:

```bash
brew install ffmpeg
```

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

## Video automation workflow

One-command generation is available for the normal path:

```bash
python3 -m modoc_pipeline run-all --input "Q&A Blog Contents List.xlsx" --row 3
```

For quick smoke tests that avoid generating every language and scene:

```bash
python3 -m modoc_pipeline run-all \
  --input "Q&A Blog Contents List.xlsx" \
  --row 3 \
  --languages english \
  --max-clips 1
```

`run-all` performs:

```text
generate -> plan-video -> veo-gemini -> render
```

The final MP4 files are written under:

```text
outputs/<run_id>/videos/
```

The individual stage commands remain useful for debugging or regenerating one
piece of a run.

After `generate` creates an `outputs/<run_id>/` folder, create a video plan:

```bash
python3 -m modoc_pipeline plan-video --run outputs/<run_id>
```

This writes:

- `video_plan.json`: high-level video settings.
- `scene_prompts.json`: 3 scenes per language by default.
- `raw_video_plan_response.txt`: raw Gemini response for debugging.
- `video_status.json`: latest video-stage status.

Generate Veo clips through Vertex AI:

```bash
python3 -m modoc_pipeline veo --run outputs/<run_id>
```

This writes clips like:

```text
outputs/<run_id>/veo/english/scene_01.mp4
outputs/<run_id>/veo/korean/scene_01.mp4
outputs/<run_id>/veo/spanish/scene_01.mp4
```

Veo execution depends on Vertex AI permissions, billing, quota, and any required
Veo allowlist. The command fails with setup instructions if local Google Cloud
credentials are missing.

Render final vertical videos:

```bash
python3 -m modoc_pipeline render --run outputs/<run_id>
```

This concatenates the Veo clips by language, generates `.srt` subtitle files,
burns subtitles into the video, and writes:

```text
outputs/<run_id>/videos/english.mp4
outputs/<run_id>/videos/korean.mp4
outputs/<run_id>/videos/spanish.mp4
```

Current audio strategy: Veo-generated audio is preserved during render. A
separate TTS stage can be added later if the team wants deterministic narration.

## Generated files

Each processed Q&A row creates a folder under `outputs/{run_id}/`:

- `source.json`: normalized source Q&A and metadata.
- `scripts.json`: English, Korean, and Spanish short-form scripts.
- `claims.json`: medical claims and evidence references.
- `review_packet.md`: reviewer-friendly source, scripts, claims, and notes.
- `status.json`: model, run status, prompt version, and generation timing.
- `raw_gemini_response.txt`: original model output for debugging.
- `video_plan.json`: high-level video settings after `plan-video`.
- `scene_prompts.json`: Veo scene prompts after `plan-video`.
- `video_status.json`: latest video stage result.

## Gemini API Veo fallback

If Vertex AI permissions are not available, you can try Veo through the Gemini
Developer API with the same `GEMINI_API_KEY` used for script generation:

```bash
python3 -m modoc_pipeline veo-gemini --run outputs/<run_id>
```

This route does not require `gcloud`, but it may still fail if the API key does
not have Veo access or if the prompt is blocked by policy. In the current SDK
mode, Gemini Developer API video generation is configured without Veo-generated
audio; Vertex remains the preferred route when generated audio is required. It
writes the same clip structure as the Vertex route, so `render` works the same
way afterward:

```bash
python3 -m modoc_pipeline render --run outputs/<run_id>
```

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
