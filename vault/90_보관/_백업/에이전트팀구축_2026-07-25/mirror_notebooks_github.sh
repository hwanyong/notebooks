#!/bin/bash
# =============================================================
# notebooks GitHub 미러 백업 (단방향: Mac → GitHub)
# 소스 1: iCloud Obsidian vault "notebooks"
# 소스 2: ~/LOCAL/03-00_STUDIES/AI (실습 코드)
# 대상  : ~/LOCAL/notebooks-mirror → github.com/hwanyong/notebooks
# 정책  : 공개 repo 전제. 민감/타인저작물 제외 + 커밋 전 2중 게이트
#         push는 자동화하지 않음 — GitHub Desktop에서 diff 검토 후 수동 Push
# 주의  : 미러·GitHub 쪽 수정은 다음 실행 때 덮어써짐 (단방향)
# =============================================================
set -euo pipefail

VAULT_SRC="$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/notebooks/"
AI_SRC="$HOME/LOCAL/03-00_STUDIES/AI/"
MIRROR="$HOME/LOCAL/notebooks-mirror"
REMOTE_URL="https://github.com/hwanyong/notebooks.git"
BRANCH="main"

# ---- 사전 점검 ----------------------------------------------
command -v git >/dev/null  || { echo "[중단] git 없음: xcode-select --install"; exit 1; }
command -v rsync >/dev/null || { echo "[중단] rsync 없음"; exit 1; }
command -v python3 >/dev/null || { echo "[중단] python3 없음 (정규화 게이트에 필요)"; exit 1; }
[ -d "$VAULT_SRC" ] || { echo "[중단] vault 경로 없음: $VAULT_SRC"; exit 1; }
[ -d "$AI_SRC" ]    || { echo "[중단] AI 경로 없음: $AI_SRC"; exit 1; }

# ---- 최초 1회 초기화 ----------------------------------------
if [ ! -d "$MIRROR/.git" ]; then
  mkdir -p "$MIRROR"
  git -C "$MIRROR" init -b "$BRANCH"
  git -C "$MIRROR" remote add origin "$REMOTE_URL"
  git -C "$MIRROR" config user.name "hwanyong"
  git -C "$MIRROR" config user.email "yoo.hwanyong@gmail.com"
  printf '.DS_Store\n' > "$MIRROR/.gitignore"
  cat > "$MIRROR/README.md" <<'EOF'
# notebooks (auto-mirror)

로컬 학습 노트의 단방향 자동 미러. 여기서 직접 편집하지 말 것 — 다음 미러 실행 때 덮어써짐.

- `vault/` — Obsidian 학습 노트 (iCloud vault 미러)
- `ai-practice/` — AI 실습 코드 (~/LOCAL/03-00_STUDIES/AI 미러)
EOF
  echo "[초기화] $MIRROR 생성 및 git init 완료"
fi

# ---- 1) vault 미러 (공개 금지 대상 제외) --------------------
rsync -a --delete \
  --exclude '.obsidian/' \
  --exclude '.DS_Store' \
  --exclude '.trash/' \
  --exclude '__temp/' \
  --exclude '고민상담/' \
  --exclude '*.deleted' \
  --exclude '*원문*' \
  --exclude '*공돌이*' \
  "$VAULT_SRC" "$MIRROR/vault/"

# ---- 2) AI 실습 미러 ----------------------------------------
rsync -a --delete \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '.DS_Store' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.git/' \
  "$AI_SRC" "$MIRROR/ai-practice/"

# ---- 정규화 내성 prune ---------------------------------------
# 이유: macOS 파일명은 NFD(자모 분해)로 저장될 수 있어, NFC로 쓴 한글
# rsync 패턴('고민상담' 등)이 바이트 비교에서 매칭 실패한다 (실증된 누출).
# 위 rsync 제외는 1차 방어로 유지하고, 여기서 NFC 정규화 비교로 확정 제거한다.
python3 - "$MIRROR" <<'PYEOF'
import os, sys, shutil, unicodedata
root = sys.argv[1]; BLOCK = ('고민상담', '원문', '공돌이'); removed = []
for dp, dns, fns in os.walk(root, topdown=True):
    if '.git' in dns: dns.remove('.git')
    for d in list(dns):
        if any(b in unicodedata.normalize('NFC', d) for b in BLOCK):
            shutil.rmtree(os.path.join(dp, d)); dns.remove(d); removed.append(os.path.join(dp, d))
    for f in fns:
        if any(b in unicodedata.normalize('NFC', f) for b in BLOCK) or f.endswith('.deleted') or f == '.env':
            os.remove(os.path.join(dp, f)); removed.append(os.path.join(dp, f))
print(f"[prune] 제거 {len(removed)}건")
for p in removed: print("  -", unicodedata.normalize('NFC', p))
PYEOF

# ---- 게이트 1: 제외 대상 잔존 검사 (NFC 정규화 비교) --------
python3 - "$MIRROR" <<'PYEOF'
import os, sys, unicodedata
root = sys.argv[1]; BLOCK = ('고민상담', '원문', '공돌이'); hits = []
for dp, dns, fns in os.walk(root):
    if '.git' in dns: dns.remove('.git')
    for n in dns + fns:
        nc = unicodedata.normalize('NFC', n)
        if any(b in nc for b in BLOCK) or n.endswith('.deleted') or n == '.env':
            hits.append(os.path.join(dp, n))
if hits:
    print("[중단] 제외 대상이 미러에 존재 — 커밋하지 않음:")
    for h in hits: print("  ", unicodedata.normalize('NFC', h))
    sys.exit(1)
print("[게이트1] 통과")
PYEOF

# ---- 게이트 2: 시크릿 패턴 스캔 (공개 repo 방어) ------------
SECRETS=$(grep -rIlE \
  'sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|AIza[0-9A-Za-z_-]{35}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY' \
  "$MIRROR" --exclude-dir=.git || true)
if [ -n "$SECRETS" ]; then
  echo "[중단] 시크릿 의심 패턴 검출 — 커밋하지 않음:"
  echo "$SECRETS"
  exit 1
fi

# ---- 커밋 (push는 GitHub Desktop에서 검토 후 수동) ----------
cd "$MIRROR"
git add -A
if git diff --cached --quiet; then
  echo "[정보] 변경 없음 — 커밋 생략"
else
  git commit -m "${COMMIT_MSG:-mirror: $(date '+%Y-%m-%d %H:%M')}"
fi
echo "[완료] HEAD = $(git rev-parse --short HEAD) — GitHub Desktop에서 diff 검토 후 Push"
