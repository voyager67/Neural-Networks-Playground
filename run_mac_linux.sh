#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
URL="http://127.0.0.1:8080"
PORT="8080"
if command -v python3 >/dev/null 2>&1; then PYTHON="python3"; elif command -v python >/dev/null 2>&1; then PYTHON="python"; else echo "Python 3 is required for the local static server." >&2; exit 1; fi
if command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1 || true; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true; fi
echo "Neural Playground is available at $URL"
echo "Keep this terminal open. Press Ctrl+C to stop."
exec "$PYTHON" -m http.server "$PORT"
