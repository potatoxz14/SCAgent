set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # Mobile-Agent-E/
cd "$HERE"
PROMPT_FILE="${PROMPT_FILE:-prompts/stage1_android.md}"
RUN_NAME="${RUN_NAME:-stage1}"
MAX_ITR="${MAX_ITR:-80}"
export ADB_PATH="${ADB_PATH:-adb}"
export BACKBONE_TYPE="${BACKBONE_TYPE:-Gemini}"

echo "== Stage-1 (Mobile-Agent-E) pre-flight =="

# 1) adb + emulator online
command -v "$ADB_PATH" >/dev/null 2>&1 || { echo "ERROR: adb not found (set ADB_PATH)."; exit 1; }
if [ "$($ADB_PATH get-state 2>/dev/null || true)" != "device" ]; then
  echo "ERROR: no online device. \`$ADB_PATH devices\`:"; "$ADB_PATH" devices
  echo "Start the emulator (or \`$ADB_PATH connect <host:port>\`) first."; exit 1
fi
echo "  device OK: $($ADB_PATH shell getprop ro.product.model 2>/dev/null | tr -d '\r')"

# 2) backbone API key present
case "$BACKBONE_TYPE" in
  Gemini) KEYVAR=GEMINI_API_KEY ;;
  OpenAI) KEYVAR=OPENAI_API_KEY ;;
  Claude) KEYVAR=CLAUDE_API_KEY ;;
  *) echo "ERROR: BACKBONE_TYPE must be Gemini/OpenAI/Claude."; exit 1 ;;
esac
[ -n "${!KEYVAR:-}" ] || { echo "ERROR: \$$KEYVAR is not set (needed for BACKBONE_TYPE=$BACKBONE_TYPE)."; exit 1; }
echo "  backbone: $BACKBONE_TYPE ($KEYVAR set)"

# 3) prompt present (fill in its 'APPS TO EXPLORE IN THIS BATCH' list before running)
[ -f "$PROMPT_FILE" ] || { echo "ERROR: prompt not found: $PROMPT_FILE"; exit 1; }
echo "  prompt : $PROMPT_FILE (passed verbatim as --instruction)"

# 4) run Mobile-Agent-E with the full prompt as the task instruction
echo "== Running Mobile-Agent-E =="
python run.py \
  --instruction "$(cat "$PROMPT_FILE")" \
  --run_name "$RUN_NAME" \
  --setting individual \
  --max_itr "$MAX_ITR"

# run.py extracts found_actions.jsonl automatically at the end of the run.
LOG_DIR="logs/${REASONING_MODEL:-gemini-2.5-flash}/mobile_agent_E/${RUN_NAME}"
cat <<EOF

== Done ==
Trajectory   : $LOG_DIR/<task_id>/steps.json  (+ screenshots/)
Found actions: $LOG_DIR/found_actions.jsonl   (auto-extracted from the trajectory)
EOF
