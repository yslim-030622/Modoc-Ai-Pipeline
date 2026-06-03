# MoDoc AI Video Pipeline

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-3.5_Flash-4285F4?logo=google&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-rendering-007808?logo=ffmpeg&logoColor=white)

MoDoc is a pediatric health platform where parents ask real questions — "my baby poops 5 times a day, is that normal?" or "which flu treatment is safer, IV or oral?" Turning those Q&A posts into short-form videos by hand took hours per video: write three scripts, generate visuals, record narration, edit, export, repeat for three languages.

This pipeline does all of it from one command. Give it a row number from the Q&A spreadsheet, and it produces three upload-ready MP4 files in English, Korean, and Spanish.

## Output Preview

> Row 24 — "How many times per day should a 2-month-old baby poop?"

<p align="center">
  <img src="docs/images/row24_english_02.jpg" width="185" alt="English scene 2" />
  &nbsp;
  <img src="docs/images/row24_korean_02.jpg" width="185" alt="Korean scene 2" />
  &nbsp;
  <img src="docs/images/row24_spanish_02.jpg" width="185" alt="Spanish scene 2" />
</p>

<p align="center">
  <img src="docs/images/row24_english_03.jpg" width="185" alt="English scene 3" />
  &nbsp;
  <img src="docs/images/row24_korean_03.jpg" width="185" alt="Korean scene 3" />
  &nbsp;
  <img src="docs/images/row24_spanish_03.jpg" width="185" alt="Spanish scene 3" />
</p>

<p align="center"><em>English &nbsp;·&nbsp; Korean &nbsp;·&nbsp; Spanish — same pipeline run, different format and character per language</em></p>

## How It Works

```bash
python3 -m modoc_pipeline generate 24
python3 -m modoc_pipeline generate 23 12 25   # multiple rows at once
```

The pipeline is a chain of agents built with LangGraph. Each agent does one job, writes its output to disk, then hands off to the next. If any step fails due to an API error or bad output, the run stops and writes a failure log. There are no retry loops or self-review stages — just a straight run from source data to finished video.

| Agent | What it does |
|-------|-------------|
| `load_source` | Reads the row from the Q&A spreadsheet |
| `grounding_agent` | Runs Google Search to verify the medical facts in the expert answer before writing anything |
| `script_writer_agent` | Writes parent-friendly scripts in English, Korean, and Spanish, staying strictly within the grounded facts |
| `persist_script_artifacts` | Saves the scripts and a clinician review packet to disk |
| `meme_planner_agent` | Researches trending formats for each language, generates three creative directions, scores and selects one, then builds a full 4-scene plan with character descriptions and image prompts |
| `image_generation_agent` | Generates scene images per language using Gemini image generation |
| `tts_agent` | Generates voiceover with Gemini TTS using language-matched voices |
| `bgm_agent` | Generates background music with Lyria using a mood prompt tuned per language |
| `render_agent` | Assembles everything with FFmpeg — Ken Burns zoom, meme text, BGM mixed at 32% under voice, 9:16 export |

## What Gets Automated

- Reading Q&A content from the Excel spreadsheet
- Verifying medical facts against Google Search before writing
- Writing scripts in three languages, constrained to what the expert answer actually says
- Picking a culturally-specific meme format (TikTok POV for English, 짤방 webtoon style for Korean, discovery-format for Spanish)
- Generating scene illustrations with a locked character design across all four scenes
- Voiceover narration with Kore (Korean), Puck (English), and Leda (Spanish)
- Background music generation with mood matched to the medical topic
- Final MP4 assembly with text overlays, audio mix, and transitions
- Per-stage timing logged for every run

## What Stays Manual

- Medical review — a clinician signs off before any video gets published
- Final visual check of the output MP4
- Publishing to YouTube Shorts, Instagram Reels, and Facebook

## Tech Stack

| Layer | Tool |
|-------|------|
| Pipeline orchestration | LangGraph |
| Script generation | Gemini 3.5 Flash + Google Search grounding |
| Image generation | Gemini image generation |
| TTS narration | Gemini TTS — Kore (Korean), Puck (English), Leda (Spanish) |
| Background music | Lyria |
| Video assembly | FFmpeg |
| Source data | Excel via openpyxl |

## Setup

```bash
git clone <repo-url> && cd Modoc-Ai-Pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
brew install ffmpeg

cp .env.example .env
# Add GEMINI_API_KEY to .env
```

## Output Files

Each run creates a folder under `logs/<run_id>/`:

```
scripts.json          — 3-language scripts
claims.json           — medical claims with evidence links
review_packet.md      — clinician review document
meme_plan.json        — scene plan with character and format decisions
meme_images/          — generated scene images per language
meme_audio/           — voiceover WAV files per scene per language
meme_bgm/             — background music per language
agent_trace.json      — per-agent status and timing for this run
```

Final videos go to `videos/Row_<n>/`:

```
Row_24_english_meme.mp4
Row_24_korean_meme.mp4
Row_24_spanish_meme.mp4
```

Timing across all runs is aggregated in `logs/run_timings.jsonl`.

---

## 한국어

MoDoc은 부모들이 실제 질문을 올리는 소아과 건강 플랫폼입니다. "아기가 하루에 5번 변을 봐요, 정상인가요?" 같은 Q&A 포스트를 숏폼 영상으로 만드는 작업을 자동화했습니다. 기존에는 영상 하나당 3개 언어 스크립트 작성, 비주얼 제작, 녹음, 편집, 내보내기까지 몇 시간이 걸렸습니다.

이 파이프라인은 명령어 하나로 전부 처리합니다. Q&A 스프레드시트의 row 번호를 입력하면 한국어, 영어, 스페인어 MP4 3개가 나옵니다.

## 명령어

```bash
python3 -m modoc_pipeline generate 24
python3 -m modoc_pipeline generate 23 12 25   # 여러 row 동시 처리
```

## 파이프라인 구조

LangGraph로 구성된 에이전트 체인입니다. 각 에이전트는 한 가지 작업만 하고, 결과를 디스크에 저장한 뒤 다음 에이전트로 넘깁니다. API 오류나 파싱 실패가 생기면 해당 지점에서 멈추고 실패 로그를 남깁니다. 재시도 루프 없이 처음부터 끝까지 직선으로 실행됩니다.

| 에이전트 | 역할 |
|---------|------|
| `load_source` | 스프레드시트에서 해당 row 읽기 |
| `grounding_agent` | 전문가 답변의 의료 정보를 Google Search로 검증 |
| `script_writer_agent` | 검증된 사실 범위 안에서 3개 언어 스크립트 작성 |
| `persist_script_artifacts` | 스크립트와 의료 검토 문서 저장 |
| `meme_planner_agent` | 언어별 트렌드 조사 후 창의적 방향 3개 생성, 점수화해서 선택, 4씬 플랜 완성 |
| `image_generation_agent` | Gemini로 언어별 씬 이미지 생성 |
| `tts_agent` | Gemini TTS로 언어별 보이스오버 생성 |
| `bgm_agent` | Lyria로 주제에 맞는 배경음악 생성 |
| `render_agent` | FFmpeg으로 이미지, 보이스오버, BGM, 밈 텍스트를 합쳐 9:16 MP4 완성 |

## 자동화된 것

- 스프레드시트에서 Q&A 콘텐츠 읽기
- 스크립트 작성 전 의료 정보 Google Search 검증
- 전문가 답변 범위 내에서 3개 언어 스크립트 작성
- 언어별 문화에 맞는 밈 포맷 선택 및 씬 설계
- 씬 간 캐릭터가 일관된 일러스트 생성
- 언어별 보이스오버 및 배경음악 생성
- 최종 MP4 조립 및 내보내기
- 스테이지별 처리 시간 자동 기록

## 수동으로 남은 것

- 의료 검토 — 의사 확인 전까지 영상 미게시
- 최종 영상 육안 확인
- YouTube Shorts, Instagram Reels, Facebook 업로드
