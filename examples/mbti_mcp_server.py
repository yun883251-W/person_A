"""Launch the local source package with the host's configured Python interpreter."""

from pathlib import Path
import sys


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mbti_mcp.server import main as run_server
    return run_server()


if __name__ == "__main__":
    raise SystemExit(main())
