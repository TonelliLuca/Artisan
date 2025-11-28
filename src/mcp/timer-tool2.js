// File: src/mcp/timer-tool2.js
import express from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import { randomUUID } from 'crypto';

const app = express();
app.use(express.json());

let sseResponse = null;
let subscribed = false;
const timers = new Map(); // id -> { name, timeoutId, seconds }

const server = new McpServer({
    name: 'timer-server',
    version: '1.0.0'
});

// -------------------------
// TOOL
// -------------------------
server.tool(
    'timerTool',
    {
        action: z.enum(["subscribe", "set"]),
        seconds: z.number().optional(),
        name: z.string().optional()
    },
    async ({ action, seconds, name }) => {

        // -------------------------
        // SUBSCRIBE
        // -------------------------
        if (action === "subscribe") {
            if (!sseResponse) {
                return { content: [{ type: 'text', text: "⚠️ SSE not connected. Start /sse from the client first." }] };
            }
            subscribed = true;
            return { content: [{ type: 'text', text: "✅ Subscription activated. You will receive events/variables." }] };
        }

        // -------------------------
        // SET TIMER
        // -------------------------
        if (action === "set") {
            if (!seconds) {
                return { content: [{ type: 'text', text: "❌ Error: missing seconds." }] };
            }

            const id = name ? String(name) : `timer-${randomUUID()}`;
            console.log(`[NODE] Timer ${id} set for ${seconds}s`);

            const timeoutId = setTimeout(() => {
                console.log(`[NODE] Timer ${id} EXPIRED → sending event`);

                if (sseResponse && subscribed) {
                    const eventData = JSON.stringify({
                        jsonrpc: "2.0",
                        method: "notifications/message",
                        params: {
                            mcpType: "event",
                            event: {
                                key: id,
                                name: "timer.finished",
                                message: `⏰ RING! Timer ${id} (${seconds}s) expired!`
                            }
                        }
                    });

                    sseResponse.write(`data: ${eventData}\n\n`);
                }

                timers.delete(id);
            }, seconds * 1000);

            // Persist timer info
            timers.set(id, { name: name || id, timeoutId, seconds });

            // Notify variable via SSE if subscribed
            if (sseResponse && subscribed) {
                const varPayload = JSON.stringify({
                    jsonrpc: "2.0",
                    method: "notifications/message",
                    params: {
                        mcpType: "variable",
                        name: id,
                        value: { name: name || id, seconds }
                    }
                });

                sseResponse.write(`data: ${varPayload}\n\n`);
            }

            // Tool response (pure text)
            return {
                content: [
                    { type: 'text', text: `⏳ Timer ${id} started for ${seconds} seconds.` },
                    {
                        type: 'text',
                        text: JSON.stringify({
                            mcpType: "variable",
                            name: id,
                            value: { name: name || id, seconds }
                        })
                    }
                ]
            };
        }

        // fallback
        return { content: [{ type: 'text', text: "Unknown action" }] };
    }
);

// -------------------------
// MCP ENDPOINT
// -------------------------
app.post('/mcp', async (req, res) => {
    const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
        enableJsonResponse: true
    });

    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
});

// -------------------------
// SSE ENDPOINT
// -------------------------
app.get('/sse', (req, res) => {
    console.log("[NODE] Nuovo client SSE connesso");

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    sseResponse = res;
    res.write(": connected\n\n");

    req.on('close', () => {
        console.log("[NODE] Client SSE disconnesso");
        sseResponse = null;
        subscribed = false;
    });
});

const PORT = 3001;

const httpServer = app.listen(PORT, () => {
    console.log(`\n🚀 Timer Server running on port ${PORT}`);
    console.log(`👉 POST http://localhost:${PORT}/mcp`);
    console.log(`👉 GET  http://localhost:${PORT}/sse\n`);
});

httpServer.on('error', (error) => {
    console.error("❌ HTTP ERROR:", error);
});

process.on('uncaughtException', err => console.error('❌ Uncaught Exception:', err));
process.on('unhandledRejection', r => console.error('❌ Unhandled Rejection:', r));
