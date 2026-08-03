# kvd_mcp/mcp_server/server.py
from fastmcp import FastMCP
from . import tools

# Create the FastMCP app
mcp = FastMCP(
    name="kvd-mcp",
    version="0.1.0",
    instructions="""
    This MCP server provides access to a local-first knowledge vault containing 
    Markdown notes. It supports hybrid search (keyword + semantic), note CRUD operations, 
    and intelligent context retrieval for AI agents.
    """
)

# Register all tools
mcp.tool()(tools.search)
mcp.tool()(tools.get_context)
mcp.tool()(tools.read_note)
mcp.tool()(tools.list_notes)
mcp.tool()(tools.create_note)
mcp.tool()(tools.update_note)
mcp.tool()(tools.delete_note)
mcp.tool()(tools.reindex_vault)
mcp.tool()(tools.get_index_stats)
mcp.tool()(tools.health_check)

# ============================================================================
# RUN THE SERVER
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if running in HTTP mode
    if "--http" in sys.argv:
        # Run via HTTP/SSE for browser-based Claude
        print("🚀 Starting kvd-mcp HTTP server on http://localhost:8100")
        print("📡 SSE endpoint: http://localhost:8100/sse")
        mcp.run(transport="sse", port=8100)
    else:
        # Run via stdio for Claude Desktop
        mcp.run(transport="stdio")