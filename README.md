# MoDoc AI Video Pipeline

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?logo=google&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-rendering-007808?logo=ffmpeg&logoColor=white)

MoDoc is a pediatric health platform where parents ask real questions — "my baby poops 5 times a day, is that normal?" or "which flu treatment is safer, IV or oral?" Turning those Q&A posts into short-form videos by hand took hours per video: write three scripts, generate visuals, record narration, edit, export, repeat for three languages.

This pipeline does all of it from one command. You give it a row number from the Q&A spreadsheet, and it produces three upload-ready MP4 files in English, Korean, and Spanish.

## Output Preview

> Row 24 — "How many times per day should a 2-month-old baby poop?"

<p align="center">
  <img src="docs/images/row24_english_01.jpg" width="185" alt="English scene 1" />
  &nbsp;
  <img src="docs/images/row24_korean_01.jpg" width="185" alt="Korean scene 1" />
  &nbsp;
  <img src="docs/images/row24_spanish_01.jpg" width="185" alt="Spanish scene 1" />
</p>

<p align="center"><em>English &nbsp;·&nbsp; Korean &nbsp;·&nbsp; Spanish — same pipeline run, culturally adapted per language</em></p>

Each language gets its own illustration style, character, and meme format. English uses a TikTok POV format, Korean uses a 짤방 webtoon style, Spanish uses a Mamá latina relatability format.

## How It Works

```bash
python3 -m modoc_pipeline generate 24
python3 -m modoc_pipeline generate 23 12 25   # multiple rows at once
```

Five stages run in sequence per row:

| Stage | What happens |
|-------|-------------|
| `scripts` | Reads the Q&A row, runs Google Search grounding for medical accuracy, writes parent-friendly scripts in all three languages |
| `meme-plan` | Picks a culturally-appropriate viral meme format per language, defines a locked character description for visual consistency |
| `imagen` | Generates 4 scene images per language using Gemini image generation, passes scene 1 as a reference image to scenes 2-4 to keep the character consistent |
| `tts + bgm` | Generates voiceover with Gemini TTS, generates background music with Lyria tuned per culture (K-pop synth-pop, hyperpop-lite, reggaeton-pop) |
| `render` | Concatenates images and audio with FFmpeg, bakes meme text, mixes BGM at 32% under the voiceover, exports 9:16 MP4 |

## What Gets Automated

- Reading Q&A content from the Excel spreadsheet
- Writing medically cautious scripts in English, Korean, and Spanish
- Researching trending meme formats per language via Google Search
- Generating culturally-adapted illustrations with a consistent character across scenes
- Voiceover narration with language-specific voices (Kore, Puck, Leda)
- Background music generation tuned per cultural context
- Final MP4 assembly with text overlays, audio mix, and fade transitions
- Stage-level timing logged per run for performance tracking

## What Stays Manual

- Medical review — a clinician signs off before any video gets published
- Publishing to YouTube Shorts, Instagram Reels, and Facebook
- Final visual check of the output MP4

## Tech Stack

| Layer | Tool |
|-------|------|
| Script generation | Gemini 2.5 Flash + Google Search grounding |
| Image generation | Gemini image generation (image-to-image for scene consistency) |
| TTS narration | Gemini TTS — Kore (Korean), Puck (English), Leda (Spanish) |
| Background music | Lyria — culturally-tuned prompts per language |
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
timing.json           — per-stage duration log for this run
```

Final videos go to `videos/Row_<n>/`:

```
Row_24_english_meme.mp4
Row_24_korean_meme.mp4
Row_24_spanish_meme.mp4
```

Timing across all runs is aggregated in `logs/run_timings.jsonl` for statistics.

---

## 한국어

MoDoc은 부모들이 실제 질문을 올리는 소아과 건강 플랫폼입니다. "아기가 하루에 5번 변을 봐요, 정상인가요?" 같은 Q&A 포스트를 숏폼 영상으로 만드는 작업을 자동화했습니다. 기존에는 영상 하나당 3개 언어 스크립트 작성, 비주얼 제작, 녹음, 편집, 내보내기까지 몇 시간이 걸렸습니다.

이 파이프라인은 명령어 하나로 전부 처리합니다. Q&A 스프레드시트의 row 번호를 입력하면 한국어, 영어, 스페인어 MP4 3개가 나옵니다.

## 명령어

```bash
python3 -m modoc_pipeline generate 24
python3 -m modoc_pipeline generate 23 12 25   # 여러 row 동시 처리
```

## 5단계 파이프라인

| 단계 | 내용 |
|------|------|
| `scripts` | Q&A row 읽기, Google Search 기반 의료 정보 검증, 3개 언어 스크립트 생성 |
| `meme-plan` | 언어별로 유행하는 밈 포맷 선택, 씬 간 캐릭터 일관성을 위한 외형 고정 |
| `imagen` | Gemini로 언어별 씬 이미지 4장 생성, scene 1 이미지를 2-4에 레퍼런스로 전달해 캐릭터 통일 |
| `tts + bgm` | Gemini TTS 보이스오버 생성, Lyria로 문화권별 BGM 생성 (K-pop, 하이퍼팝, 레게톤) |
| `render` | FFmpeg으로 이미지, 오디오, 밈 텍스트, BGM을 합쳐 9:16 MP4 완성 |

## 자동화된 것

- 스프레드시트에서 Q&A 콘텐츠 읽기
- 의료적으로 검증된 3개 언어 스크립트 작성
- 언어별 유행 밈 포맷 리서치 및 적용
- 씬 간 캐릭터가 일관된 일러스트 생성
- 언어별 보이스오버, 배경음악 생성
- 최종 MP4 조립 및 내보내기
- 스테이지별 처리 시간 자동 기록

## 수동으로 남은 것

- 의료 검토 — 의사 확인 전까지 영상 미게시
- YouTube Shorts, Instagram Reels, Facebook 업로드
- 최종 영상 육안 확인
