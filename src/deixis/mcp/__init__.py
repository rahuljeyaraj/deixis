"""The core over MCP (the `mcp` extra): `deixis-mcp`, the stdio server, and `DeixisClient`,
a small synchronous client with a direct (in-process) fallback.

Only this subpackage imports the MCP SDK; `import deixis` never does.
"""

from deixis.mcp.client import DeixisClient

__all__ = ["DeixisClient"]
