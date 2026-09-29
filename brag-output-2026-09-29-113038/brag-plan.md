# Brag Plan: FIE — detailed explainer (~2:06)

## Goal
A longer, easy-to-follow walkthrough of what FIE does, how the input pipeline works step by step, what happens after the model answers, and the honest results.

## Tone
- Preset: polished
- Direction: "professional and simple to understand"; plain words, one idea per scene
- Format: 1920x1080 · Duration: 126s (voice sets the pace)

## Sources
`docs/FACT_SHEET.md` (all numbers), `docs/ARCHITECTURE.md` (pipeline, weights, boosts, routing, output side), `README.md` (architecture diagram), `deploy/huggingface/space_app.py` (demo prompts), live `scan_prompt()` output.

## Storyboard (scene · start · what's shown)
1. Hook · 0.00 — "Can your AI be tricked?" + attack prompt typing
2. Problem · 6.09 — 4 attack cards: hidden instructions, fiction wrapping, other languages, strange (GCG-style) text
3. What FIE is · 15.27 — Users → FIE → Your model; `@monitor()`
4. Step 1 · 23.90 — memory/hash check: known attack → block, known safe → allow, else → step 2
5. Step 2 · 30.63 — 12 layers in 3 groups (Patterns / Structure / Meaning), run in parallel
6. Step 3 · 44.44 — weighted vote (real weights), agreement boost (+0.08/+0.12), session boost (≤+0.20), router: low → allow, high → block, borderline → LlamaGuard
7. Real scan · 63.63 — actual scan_prompt output: regex 0.82, prompt_guard 0.86 → BLOCKED prompt extraction, 33 ms avg
8. After the answer · 73.82 — output scanner (leak/harm/echo), server: 3 shadow models → 434 signals → XGBoost → 3-agent jury → validated/corrected/blocked; feedback memory
9. Results · 90.17 — 85.8% macro recall on 4 benchmarks; 23M params, offline core scan, 33 ms
10. Tested honestly · 99.44 — 52.5% JailbreakBench leak removed; over-refusal 53.6% XSTest, 90.4% OR-Bench-hard; 20B guard 80%
11. Outro · 118.16 — logo, pip install fie-sdk, live demo

## Voiceover script (Kokoro af_heart, 1.08x) — see `vo-script.txt`

## Audio
Music: happy-beats-business-moves-vol-1 at 0.12 under voice, lift + fade at end. SFX: typing ticks, soft drops on attack cards, soft impact on verdict, bell on logo. Audio-reactive backdrop glow from music bass.

## Claims policy
Numbers only from the fact sheet. The borderline LlamaGuard second opinion is an optional network call, so "offline" is stated for the core scan only.
