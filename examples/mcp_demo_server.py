"""无副作用的 stdio MCP 联调服务。"""

from __future__ import annotations

import sys
from typing import Annotated, Any

from pydantic import Field

FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


def create_server():
    """Create the server without starting it during import."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    server = FastMCP(
        "Person A MCP demo",
        instructions="A deterministic, side-effect-free arithmetic tool for integration tests.",
    )

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        )
    )
    def add_numbers(a: FiniteNumber, b: FiniteNumber) -> dict[str, Any]:
        """Add two finite numbers without accessing files, network, or persistent state."""
        total = a + b
        return {"sum": total, "text": f"{a:g} + {b:g} = {total:g}"}

    return server


def main() -> int:
    try:
        server = create_server()
    except Exception:
        print(
            "demo_mcp_startup_failed: install the project's MCP optional dependencies.",
            file=sys.stderr,
        )
        return 1
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
