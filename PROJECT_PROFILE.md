# Project Profile — Kids AI Studio (kidvid)

> Turn one sentence into a finished animated cartoon or sing-along song for young children — free, using open-source AI models.

## 1. Business Overview
- **Owner / Brand:** Jordan Design Hub (JD Hub) — Mubogi Gastavas Jordan Tech Ecosystem
- **Contact:** jordandesignhub@gmail.com · WhatsApp +256 754 687 597
- **Category:** EdTech / Creator Tools / Media & AI Solutions
- **Status:** prototype
- **Links:** https://github.com/Mubogi/kids-ai-studio · Kaggle notebook (pending publish)

## 2. Problem & Target Market
- **Problem:** AI video tools cost $20–50/month and are priced per clip, so Ugandan schools, parents, churches and small content creators cannot afford to produce children's story animations or nursery-rhyme videos at volume.
- **Target users:** Schools and nurseries, parents and homeschoolers, children's content creators, church children's ministries, JD Hub Academy students learning AI.
- **Market context:** Runs free on a Kaggle GPU (~30 h/week), so production cost is effectively UGX 0 beyond data. Fits a market where monthly USD subscriptions are hard to justify but mobile data is affordable. Everything works from a browser, no local GPU needed.

## 3. Value Proposition & Features
- One prompt → animated story video with 4-beat story arc (setup, journey, problem, resolution)
- Sing-along song mode with generated lyrics and large readable captions
- Image-to-video: use your own photos or drawings as the starting frame per scene
- Music generated per video (ACE-Step, Apache-2.0) with optional narration
- Storyboard preview before spending any GPU time — no wasted quota
- Runs on a free Kaggle T4; no subscription, no per-clip pricing
- JD Hub yellow branding burned into title card and captions
- Swappable model backends, so newer open models can be dropped in later
- Offline test path: whole pipeline runs with no GPU for training and demos
- Fully permissive licensing (Apache-2.0 / MIT), outputs may be sold

## 4. Business / Monetization Model
- **Pricing:** freemium — free self-serve tool
- **Revenue streams:** paid custom episode production for schools and churches; JD Hub Academy AI course upsell; setup/training fees; optional Pro unlock for batch rendering and custom branding
- **Payment methods:** MTN MoMo, Airtel Money, Flutterwave

## 5. Tech Stack
| Layer | Tech |
|-------|------|
| Backend | Python — kidvid library, ffmpeg for assembly |
| Frontend | Generated branded HTML preview page (standalone, no build step) |
| Database | None — filesystem outputs (MP4, storyboard JSON) |
| Models | Wan 2.2 (Apache-2.0), LTX-Video, ACE-Step, MusicGen |
| Mobile/Desktop | n/a — browser-based |
| Deploy | Kaggle Notebooks (free GPU T4 x2), optionally RunPod/Colab |

## 6. Roadmap & Status
- **Current milestone:** End-to-end working — prompt to branded MP4 with music, narration and captions, verified locally with the no-GPU path.
- **Next steps:** Confirm the WanGP entrypoint on a live Kaggle T4; add an LLM-written storyboard for richer stories; batch mode to render a full episode series; Swahili and Luganda narration voices.
- **Known gaps:** Free T4 cannot run full-precision Wan 14B or HunyuanVideo; template stories are formulaic without an LLM; music is not beat-synced to picture.

## 7. Metrics (optional)
- Videos generated: n/a (pre-production)
- Last updated: 2026-09-22

---
*Template version: 1.0 — kept identical across all JD Hub projects. Update only the content, not the structure.*
