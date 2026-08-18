"""thor-mcp — MCP server for programmatic benchmarking and model management.

Exposes tools (benchmark.run, models.optimize, ...), resources
(thor://benchmarks/results, ...) and prompts to MCP clients such as
Claude Desktop, Cursor and LangChain agents.

Run modes:

* stdio: ``thor-mcp --stdio`` (MCP protocol over stdio)
* http:  ``thor-mcp --http --port 3000`` (FastAPI REST bridge)
"""

__version__ = "0.1.0"
