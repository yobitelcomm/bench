#!/usr/bin/env bash
# Round 18.D — validate voice.transcription.librispeech-clean-mini on TestBM.
#
# Single H100 + faster-whisper-server + bench. End-to-end signed envelope
# produced on real audio, then SCP'd back to the local workstation and
# optionally published to HF as `Yobitel/voice-asr-validation`.
#
# Follows the team convention (memory/feedback_benchmark_workflow.md):
#   * local orchestration, remote execution
#   * full teardown: rm bench/venv + clear HF cache + truncate bash/zsh history
#   * leave-no-trace on the remote box
#
# Prereqs (on the local box where you run THIS script):
#   * SSH alias `TestBM` configured for taiga-support@2a0d:d40:0:107d::1c
#   * `HF_TOKEN` exported (only needed if you pass --publish)
#   * docker available on the remote box (for faster-whisper-server)
#
# Usage:
#   scripts/testbm_voice_validation.sh             # signed envelope, no publish
#   scripts/testbm_voice_validation.sh --publish   # also push to Yobitel/voice-asr-validation
#   scripts/testbm_voice_validation.sh --dry-run   # print everything, do nothing

set -euo pipefail

# -------- knobs ----------------------------------------------------------- #
REMOTE_HOST="${REMOTE_HOST:-TestBM}"            # SSH alias preferred
REMOTE_USER="${REMOTE_USER:-taiga-support}"      # fallback if no alias
REMOTE_IPV6="${REMOTE_IPV6:-2a0c:3c80:1:8::102}"
WHISPER_MODEL="${WHISPER_MODEL:-Systran/faster-whisper-large-v3}"
WHISPER_PORT="${WHISPER_PORT:-8000}"
HF_TARGET_REPO="${HF_TARGET_REPO:-Yobitel/voice-asr-validation}"
REMOTE_WORK="${REMOTE_WORK:-\$HOME/bench-voice-validation}"
LOCAL_OUT="${LOCAL_OUT:-$HOME/bench-voice-envelopes}"

PUBLISH=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --publish) PUBLISH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

ssh_run() {
  # Use the alias if it exists; otherwise build a direct IPv6 SSH command.
  if ssh -G "$REMOTE_HOST" >/dev/null 2>&1; then
    run ssh -o ConnectTimeout=10 "$REMOTE_HOST" "$@"
  else
    run ssh -o ConnectTimeout=10 -o AddressFamily=inet6 \
      "${REMOTE_USER}@${REMOTE_IPV6}" "$@"
  fi
}

scp_back() {
  local remote_path="$1"
  local local_path="$2"
  if ssh -G "$REMOTE_HOST" >/dev/null 2>&1; then
    run scp "${REMOTE_HOST}:${remote_path}" "$local_path"
  else
    run scp -o AddressFamily=inet6 \
      "[${REMOTE_USER}@${REMOTE_IPV6}]:${remote_path}" "$local_path"
  fi
}

mkdir -p "$LOCAL_OUT"

# -------- step 1: stage repo + start whisper server ---------------------- #
echo "==> [1/5] stage repo on TestBM + start ${WHISPER_MODEL} on :${WHISPER_PORT}"
ssh_run "set -e
  rm -rf ${REMOTE_WORK}
  mkdir -p ${REMOTE_WORK}
  cd ${REMOTE_WORK}
  git clone --depth=1 https://github.com/yobitelcomm/bench.git
  cd bench
  # Python 3.12 is required by inferencebench but TestBM defaults to 3.10 —
  # install via uv which manages its own toolchain. (Per memory/learnings.md.)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  uv sync --all-packages --dev --prerelease=allow

  # Start faster-whisper-server in the background; kill any stale one first.
  # sudo because taiga-support isn't in the docker group (we don't change that).
  # --network host is REQUIRED on TestBM: the image runs 'uv run' at startup
  # which re-resolves deps from PyPI, and the default docker bridge network
  # can't do DNS on this node. Host networking gives it the host's resolver
  # (and makes the server reachable on localhost:${WHISPER_PORT} directly).
  sudo docker rm -f voice-fwserver 2>/dev/null || true
  sudo docker run -d --name voice-fwserver --gpus all --network host \
    -e WHISPER__MODEL=${WHISPER_MODEL} -e WHISPER__COMPUTE_TYPE=float16 \
    fedirz/faster-whisper-server:latest-cuda
"

# -------- step 2: wait for server to be ready --------------------------- #
echo "==> [2/5] wait for whisper server to accept POSTs (up to 90s)"
ssh_run "for i in \$(seq 1 30); do
    if curl -sf -o /dev/null -m 2 http://localhost:${WHISPER_PORT}/health; then
      echo \"server ready in \${i}*3 s\"; break
    fi
    sleep 3
  done"

# -------- step 3: mint dev key + run the benchmark ---------------------- #
echo "==> [3/5] run voice.transcription.librispeech-clean-mini, dev-key signed"
ssh_run "set -e
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  cd ${REMOTE_WORK}/bench
  uv run python -c 'from inferencebench.envelope import generate_dev_keypair; generate_dev_keypair(\"cosign.key\")'
  uv run bench run voice.transcription.librispeech-clean-mini \
    --model ${WHISPER_MODEL} --engine whisper-http \
    --base-url http://localhost:${WHISPER_PORT}/v1 \
    --signing-mode dev --dev-key cosign.key \
    --output ./envelopes
  ls -la ./envelopes/"

# -------- step 4: scp envelope back + (optional) publish --------------- #
echo "==> [4/5] copy envelope back to ${LOCAL_OUT}/"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
# bench writes the envelope as <content-hash>.json (+ a samples-<ts>.jsonl
# diagnostic). Resolve its ABSOLUTE path on the remote first — scp/SFTP does
# not expand `$HOME`, `~`, or globs in the remote path, so we expand them in
# the remote shell via ssh and scp the literal absolute path back.
REMOTE_ENV="$(ssh_run "ls ${REMOTE_WORK}/bench/envelopes/*.json | head -1" | tr -d '\r')"
if [[ "$DRY_RUN" -eq 0 && -z "$REMOTE_ENV" ]]; then
  echo "FATAL: no envelope found on remote under ${REMOTE_WORK}/bench/envelopes/" >&2
  exit 1
fi
scp_back "${REMOTE_ENV}" "${LOCAL_OUT}/voice-asr-${TS}.json"

if [[ "$PUBLISH" -eq 1 ]]; then
  echo "==> publish to hf://datasets/${HF_TARGET_REPO}"
  : "${HF_TOKEN:?HF_TOKEN must be set when --publish is passed}"
  run python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo(repo_id='${HF_TARGET_REPO}', repo_type='dataset', exist_ok=True)
api.upload_file(
    path_or_fileobj='${LOCAL_OUT}/voice-asr-${TS}.json',
    path_in_repo='voice-asr-${TS}.json',
    repo_id='${HF_TARGET_REPO}', repo_type='dataset',
    commit_message='voice.transcription.librispeech-clean-mini on H100 + faster-whisper-server',
)
print('uploaded')
"
fi

# -------- step 5: teardown ---------------------------------------------- #
echo "==> [5/5] teardown — rm bench dir, clear HF cache, truncate shell history"
ssh_run "set -e
  sudo docker rm -f voice-fwserver >/dev/null 2>&1 || true
  rm -rf ${REMOTE_WORK}
  rm -rf \$HOME/.cache/huggingface \$HOME/.cache/uv
  rm -f \$HOME/.local/bin/uv \$HOME/.local/bin/uvx 2>/dev/null || true
  : > \$HOME/.bash_history 2>/dev/null || true
  : > \$HOME/.zsh_history 2>/dev/null || true
  history -c 2>/dev/null || true
  echo 'teardown OK'"

echo "==> done. envelope at ${LOCAL_OUT}/voice-asr-${TS}.json"
echo "    inspect with: uv run bench summary ${LOCAL_OUT}/voice-asr-${TS}.json"
