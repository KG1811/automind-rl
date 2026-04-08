#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$(pwd)}"
PING_URL="${PING_URL:-http://127.0.0.1:8000/health}"
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-900}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log() {
  printf "%b\n" "$1"
}

pass() {
  printf "${GREEN}[PASS]${NC} %s\n" "$1"
}

fail() {
  printf "${RED}[FAIL]${NC} %s\n" "$1"
}

hint() {
  printf "${YELLOW}[HINT]${NC} %s\n" "$1"
}

stop_at() {
  printf "\n${RED}${BOLD}Validation stopped at %s${NC}\n" "$1"
  exit 1
}

run_with_timeout() {
  local timeout_seconds="$1"
  shift
  timeout "$timeout_seconds" "$@"
}

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD} AutoMind Pre-Submission Validation${NC}\n"
printf "${BOLD}========================================${NC}\n\n"

log "${BOLD}Step 1/3: Checking environment reset health${NC} ..."

if ! command -v curl >/dev/null 2>&1; then
  fail "curl command not found"
  hint "Install curl before running this script."
  stop_at "Step 1"
fi

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$PING_URL" || true)
if [ "$HTTP_CODE" = "200" ]; then
  pass "Environment responded with HTTP 200"
else
  fail "Environment health endpoint returned HTTP $HTTP_CODE (expected 200)"
  hint "Start the local server with: uvicorn main:app --host 0.0.0.0 --port 8000"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  hint "Install it with: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  if [ -n "$VALIDATE_OUTPUT" ]; then
    log "  $VALIDATE_OUTPUT"
  fi
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n\n"
