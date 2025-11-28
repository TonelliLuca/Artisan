# Artisan

## Quick test 

Prerequisites:
- Node.js (>=16) and npm
- Java (11+ or project-required version) and Maven (or Gradle)

1) Start the MCP server (in a terminal)
- Open terminal at project root and run:
  cd src/mcp
  npm install
  npm run timer
- The server listens on port 3001 (SSE at /sse and MCP at /mcp). Keep this terminal open.

2) Run the Java tests

Notes:
- Integration tests will skip automatically if the MCP server is not reachable on localhost:3001.
- Keep the MCP server running while tests execute (Ctrl+C to stop it).
