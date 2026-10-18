"""使用宿主指定的 Python 启动内置浏览器 MCP；导入本文件不会运行服务。"""

from pathlib import Path
import sys


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from browser_mcp.server import main as run_server

    return run_server()


if __name__ == "__main__":
    raise SystemExit(main())
