#!/usr/bin/env bash
# 一键运行全部分析并生成图表
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "未找到 .venv，请先按 README 创建虚拟环境并安装依赖" >&2
    exit 1
fi

"$PYTHON" knn_iris_analysis.py "$@"
"$PYTHON" knn_from_scratch.py
