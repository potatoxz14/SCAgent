# Stage 1 — Sensitive-Event Discovery (Android emulator)

An LLM agent (Mobile-Agent-E) explores mobile apps on an **Android emulator** and
discovers *sensitive actions* — UI taps whose identity alone leaks a private fact.
The reconnaissance instruction is in [`prompts/stage1_android.md`](prompts/stage1_android.md).

## Prerequisites
- An Android **emulator** online over `adb` (`adb get-state` → `device`).
- `pip install -r requirements.txt`.
- A backbone API key (default backbone is Gemini):
  ```bash
  export BACKBONE_TYPE=Gemini        # or OpenAI / Claude
  export GEMINI_API_KEY=...          # OPENAI_API_KEY / CLAUDE_API_KEY for the others
  ```

## Run
1. In `prompts/stage1_android.md`, fill in the app list under
   **APPS TO EXPLORE IN THIS BATCH**.
2. Launch:
   ```bash
   bash scripts/run_task.sh
   ```

