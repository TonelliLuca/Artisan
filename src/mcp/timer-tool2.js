
import express from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import { randomUUID } from 'crypto';

const app = express();
app.use(express.json());

let sseResponse = null;
// track subscriptions per UUID (RFC4122 string)
const subscribedUuids = new Set();
// queue notifications per-UUID while SSE is not connected
const pendingNotifications = new Map();
// timers map: id -> { name, timeoutId, seconds }
const timers = new Map();

const server = new McpServer({
    name: 'timer-server',
    version: '1.0.0'
});

function now() {
    return new Date().toISOString();
}

function log(...args) {
    console.log(now(), ...args);
}

// helper to enqueue a notification string for a uuid
function enqueueNotification(uuid, payload) {
    if (!pendingNotifications.has(uuid)) pendingNotifications.set(uuid, []);
    pendingNotifications.get(uuid).push(payload);
}

// flush pending notifications for all subscribed UUIDs (called when SSE connects)
function flushPendingNotifications() {
    if (!sseResponse) return;
    for (const [uuid, list] of pendingNotifications.entries()) {
        if (!subscribedUuids.has(uuid)) continue; // only deliver for explicitly subscribed UUIDs
        for (const payload of list) {
            log('[SSE][FLUSH] Sending queued payload for UUID:', uuid);
            try {
                sseResponse.write(`data: ${payload}\n\n`);
            } catch (e) {
                log('[SSE][FLUSH] Failed to write payload for UUID', uuid, e);
            }
        }
        pendingNotifications.delete(uuid);
    }
}

// -------------------------
// TOOL
// -------------------------
server.tool(
    'timerTool',
    {
        action: z.enum(["subscribe", "set"]),
        seconds: z.number().optional(),
        name: z.string().optional(),
        uuid: z.string().uuid()
    },
    async ({ action, seconds, name, uuid }) => {

        log('[TOOL] Called timerTool', { action, seconds, name, uuid, sseConnected: !!sseResponse, subscribedForUuid: subscribedUuids.has(uuid) });

        // -------------------------
        // SUBSCRIBE
        // -------------------------
        if (action === "subscribe") {
            subscribedUuids.add(uuid);
            log('[TOOL][SUBSCRIBE] UUID subscribed:', uuid, 'sseConnected=', !!sseResponse);

            const eventPayloadObj = {
                jsonrpc: "2.0",
                method: "notifications/message",
                params: {
                    uuid,
                    mcpType: "event",
                    event: {
                        key: "subscription-ack",
                        name: "subscription.started",
                        message: "Subscription successfully activated via SSE"
                    }
                }
            };
            const eventPayload = JSON.stringify(eventPayloadObj);

            if (sseResponse) {
                log('[TOOL][SUBSCRIBE] Sending SSE ack immediately for UUID:', uuid);
                try {
                    sseResponse.write(`data: ${eventPayload}\n\n`);
                } catch (e) {
                    log('[TOOL][SUBSCRIBE] Failed to write SSE ack, queuing instead', uuid, e);
                    enqueueNotification(uuid, eventPayload);
                }
            } else {
                log('[TOOL][SUBSCRIBE] SSE not connected yet; queuing ack for UUID:', uuid);
                enqueueNotification(uuid, eventPayload);
            }

            // 4. Risposta standard del Tool (Testo)
            const msg = {
                uuid,
                content: [{ type: 'text', text: "✅ Subscription activated. You will receive events and variables via SSE." }]
            };

            return msg;
        }

        // -------------------------
        // SET TIMER
        // -------------------------
        if (action === "set") {
            if (!seconds) {
                log('[TOOL][SET] Error: missing seconds. UUID:', uuid);
                return { uuid, content: [{ type: 'text', text: "❌ Error: missing seconds." }] };
            }

            const id = name ? String(name) : `timer-${randomUUID()}`;
            log('[TOOL][SET] Timer set', { id, seconds, uuid });

            const timeoutId = setTimeout(() => {
                log('[NODE] Timer expired', { id, seconds, uuid, sseConnected: !!sseResponse, subscribedForUuid: subscribedUuids.has(uuid) });

                const eventDataObj = {
                    jsonrpc: "2.0",
                    method: "notifications/message",
                    params: {
                        uuid,
                        mcpType: "event",
                        event: {
                            key: id,
                            name: "timer.finished",
                            message: `⏰ RING! Timer ${id} (${seconds}s) expired!`
                        }
                    }
                };
                const eventData = JSON.stringify(eventDataObj);

                if (sseResponse && subscribedUuids.has(uuid)) {
                    log('[NODE] Sending SSE event for UUID:', uuid, 'eventKey:', id);
                    try {
                        sseResponse.write(`data: ${eventData}\n\n`);
                    } catch (e) {
                        log('[NODE] Failed to send SSE event for UUID', uuid, e);
                        // fallback to queue the event
                        enqueueNotification(uuid, eventData);
                    }
                } else {
                    // either SSE not connected or this uuid didn't subscribe => queue only if uuid subscribed
                    if (subscribedUuids.has(uuid)) {
                        log('[NODE] SSE not connected - queuing event for UUID:', uuid);
                        enqueueNotification(uuid, eventData);
                    } else {
                        log('[NODE] UUID not subscribed - NOT sending nor queuing event for UUID:', uuid);
                    }
                }

                timers.delete(id);
            }, seconds * 1000);

            // Persist timer info
            timers.set(id, { name: name || id, timeoutId, seconds });

            // Prepare the variable payload that describes the timer
            const varPayloadObj = {
                jsonrpc: "2.0",
                method: "notifications/message",
                params: {
                    uuid,
                    mcpType: "variable",
                    name: id,
                    value: { name: name || id, seconds }
                }
            };
            const varPayload = JSON.stringify(varPayloadObj);

            // Notify variable via SSE if subscribed for this uuid; otherwise queue if subscribed but no SSE
            if (sseResponse && subscribedUuids.has(uuid)) {
                log('[TOOL][SET] Sending SSE variable for UUID:', uuid, 'key:', id);
                try {
                    sseResponse.write(`data: ${varPayload}\n\n`);
                } catch (e) {
                    log('[TOOL][SET] Failed to write SSE variable for UUID', uuid, e);
                    enqueueNotification(uuid, varPayload);
                }
            } else {
                if (subscribedUuids.has(uuid)) {
                    log('[TOOL][SET] SSE not connected - queuing variable for UUID:', uuid);
                    enqueueNotification(uuid, varPayload);
                } else {
                    log('[TOOL][SET] UUID did NOT subscribe - variable not sent/queued for UUID:', uuid);
                }
            }

            // Tool response (pure text) — include JSON payload in one text element so callers that parse it can use it
            const response = {
                uuid,
                content: [
                    { type: 'text', text: `⏳ Timer ${id} started for ${seconds} seconds.` },
                    {
                        type: 'text',
                        // also include the uuid inside the JSON payload so callers that parse it can see it
                        text: JSON.stringify({
                            uuid,
                            mcpType: "variable",
                            name: id,
                            value: { name: name || id, seconds }
                        })
                    }
                ]
            };

            log('[TOOL][RESPONSE] Returning tool response for UUID:', uuid, 'response content:', response.content.map(c => (c.text ? c.text : c)));
            return response;
        }

        // fallback
        log('[TOOL] Unknown action', action);
        return { uuid, content: [{ type: 'text', text: "Unknown action" }] };
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
    log("[NODE] New SSE client connected");

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    sseResponse = res;
    log('[SSE] SSE stream established. subscribedUuidsCount=', subscribedUuids.size);
    res.write(": connected\n\n");

    // When a client connects, flush queued notifications for subscribed UUIDs
    flushPendingNotifications();

    req.on('close', () => {
        log("[NODE] SSE client disconnected");
        sseResponse = null;
        // do not clear subscribedUuids: subscriptions are per-UUID and persist until explicitly removed
    });
});

const PORT = 3001;

const httpServer = app.listen(PORT, () => {
    log(`🚀 Timer Server running on port ${PORT}`);
    log(`👉 POST http://localhost:${PORT}/mcp`);
    log(`👉 GET  http://localhost:${PORT}/sse`);
});

httpServer.on('error', (error) => {
    log("❌ HTTP ERROR:", error);
});

process.on('uncaughtException', err => log('❌ Uncaught Exception:', err));
process.on('unhandledRejection', r => log('❌ Unhandled Rejection:', r));