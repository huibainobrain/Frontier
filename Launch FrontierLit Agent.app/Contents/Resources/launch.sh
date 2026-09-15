#!/bin/bash
# Launcher logic for the FrontierLit Agent web demo, invoked via
# Contents/Resources/launch.sh by the AppleScript app stub in
# Contents/MacOS (see launcher.applescript / osacompile — a raw shell
# script can't be CFBundleExecutable directly: LaunchServices can't
# read an architecture out of a shebang script and wrongly prompts to
# install Rosetta, so a real (osacompile-built) app stub calls this
# instead via `do shell script`).
#
# Resolves everything relative to this script's own location (NOT a
# hardcoded path) so the whole project folder keeps working after
# being zipped, moved, or unzipped on a different Mac.

# Contents/Resources/launch.sh -> up 3 levels -> the .app's parent,
# i.e. the project root this .app was placed in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PORT=8000
URL="http://127.0.0.1:${PORT}"
LOG_FILE="$PROJECT_ROOT/data/web_ui_server.log"

notify() {
    osascript -e "display notification \"$1\" with title \"FrontierLit Agent\"" >/dev/null 2>&1
}

fail_dialog() {
    osascript -e "display dialog \"$1\" with title \"FrontierLit Agent 启动失败\" buttons {\"OK\"} default button 1 with icon caution" >/dev/null 2>&1
    # Exit 0 deliberately: the dialog above already told the user what
    # happened. A nonzero exit here would also surface AppleScript's
    # own generic "do shell script" error dialog on top of it.
    exit 0
}

server_up() {
    code=$(curl -s -m 2 -o /dev/null -w "%{http_code}" "$URL/" 2>/dev/null)
    [ "$code" = "200" ]
}

# Already running (e.g. double-clicked twice) -> just refocus the tab.
if server_up; then
    open "$URL"
    exit 0
fi

if [ ! -x "$PROJECT_ROOT/.venv/bin/uvicorn" ]; then
    fail_dialog "还没有找到运行环境（.venv）。请先在终端里进入项目目录，执行一次安装：\n\nuv venv --python 3.11 && source .venv/bin/activate && uv pip install -e \\\".[dev]\\\"\n\n完成后再双击本 App。"
fi

if [ ! -f "$PROJECT_ROOT/.env" ] || ! grep -q "^GOOGLE_API_KEY=.\+" "$PROJECT_ROOT/.env" 2>/dev/null; then
    fail_dialog "还没有配置 Gemini API Key。请在项目目录下创建 .env 文件（可从 .env.example 复制），并填入 GOOGLE_API_KEY，然后再双击本 App。"
fi

mkdir -p "$PROJECT_ROOT/data"
notify "正在启动，请稍候…"

cd "$PROJECT_ROOT"
nohup "$PROJECT_ROOT/.venv/bin/uvicorn" app:app --host 127.0.0.1 --port "$PORT" > "$LOG_FILE" 2>&1 &
disown

# Poll up to ~30s for the server to come up.
ready=0
for _ in $(seq 1 30); do
    if server_up; then
        ready=1
        break
    fi
    sleep 1
done

if [ "$ready" != "1" ]; then
    if grep -q "GOOGLE_API_KEY is required" "$LOG_FILE" 2>/dev/null; then
        fail_dialog "Gemini API Key 无效或缺失。请检查项目目录下的 .env 文件中的 GOOGLE_API_KEY，然后再试一次。"
    fi
    fail_dialog "服务启动超时。详细日志见：\n$LOG_FILE"
fi

open "$URL"
