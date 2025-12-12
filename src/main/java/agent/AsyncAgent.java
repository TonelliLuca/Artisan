package agent;

import agent.activity.Activity;
import agent.activity.ReasoningStep;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.NullNode;
import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.CopyOnWriteArrayList;

public class AsyncAgent<T extends ReactBrain> {
    private final OpenAiChatModel model;
    private final Class<T> agentInterface;
    private final T agentBrain;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AllMiniLmL6V2EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    private final UUID agentId = UUID.randomUUID();
    private final Logger logger = LoggerFactory.getLogger(AsyncAgent.class);

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final Map<String, Map<String, JsonNode>> variablesPerActivity = new ConcurrentHashMap<>();
    private final Map<String, List<JsonNode>> eventsPerActivity = new ConcurrentHashMap<>();
    private final Map<String, Activity> pendingActivities = new ConcurrentHashMap<>();

    private final String sseUrl;

    private final BlockingQueue<Activity> activityQueue = new LinkedBlockingQueue<>();
    private final AtomicBoolean loopRunning = new AtomicBoolean(true);

    private static final int WINDOW_SIZE = 5;

    private AsyncAgent(Builder<T> builder) {
        this.model = builder.model;
        this.agentInterface = builder.agentInterface;
        this.sseUrl = builder.sseUrl;

        var agent = AgenticServices
                .agentBuilder(agentInterface)
                .chatModel(model)
                .beforeAgentInvocation(request -> logger.debug("[BEFORE AGENT] {}", request))
                .afterAgentInvocation(response -> logger.debug("[AFTER AGENT] {}", response));

        if (builder.tools != null && builder.tools.length > 0) {
            agent.tools(builder.tools);
        }

        if (builder.mcpToolProvider != null) {
            agent.toolProvider(builder.mcpToolProvider);
        }

        this.agentBrain = agent.build();

        // start SSE listener if provided
        if (this.sseUrl != null && !this.sseUrl.isEmpty()) {
            startSseListener();
        }

        // start event loop
        executor.submit(this::eventLoop);
    }

    // Enqueue a new activity (goal = request)
    public void request(String request) {
        if (request == null || request.isBlank()) return;
        Activity activity = new Activity(request);
        activityQueue.offer(activity);
        logger.info("Queued Activity {} (goal={})", activity.getUuid(), request);
    }


    private void eventLoop() {
        logger.info("🚦 Agent event loop started");
        while (loopRunning.get() && !Thread.currentThread().isInterrupted()) {
            try {
                Activity activity = activityQueue.poll(500, TimeUnit.MILLISECONDS);
                if (activity == null) continue;
                if (activity.isCompleted()) {
                    logger.debug("Skipping completed activity {}", activity.getUuid());
                    continue;
                }
                Activity.Status status = activity.getStatus();
                String phase = status == null ? "UNKNOWN" : status.name();
                logger.debug("Processing activity {} phase={}", activity.getUuid(), phase);

                // prepare lastStep and context payloads expected by ReactBrain methods
                String history = this.extractActivityHistory(activity, WINDOW_SIZE);
                // compute activityUuid early so snapshots can be tied to the correct activity
                String activityUuid = activity.getUuid().toString();
                String contextJson;
                try {
                    Map<String, Object> ctx = new HashMap<>();
                    // include the activity UUID so tools can correlate calls; also provide a per-activity variables/events window
                    ctx.put("activityUuid", activityUuid);
                    ctx.put("variables", snapshotVariables(activityUuid)); // variables scoped to this activity
                    ctx.put("events", eventsPerActivity.getOrDefault(activityUuid, Collections.emptyList()));
                    contextJson = objectMapper.writeValueAsString(ctx);
                } catch (Exception e) {
                    logger.debug("Failed to serialize context, using empty context", e);
                    contextJson = "";
                }

                switch (status) {
                    case REASONING -> {
                        String reasoningResult = invokeAgentMethod("reason", activity.getGoal(), history, contextJson);
                        Map<String, Object> snapshot = snapshotVariables(activityUuid);
                        activity.addStep(new ReasoningStep("reason", activity.getGoal(), reasoningResult, snapshot));
                        activity.setStatus(Activity.Status.ACTION);
                        logger.info("Activity {} moved to ACTION", activity.getUuid());
                        activityQueue.offer(activity);
                    }
                    case ACTION -> {
                        List<JsonNode> currentEvents = eventsPerActivity.get(activityUuid);
                        int initialEventCount = (currentEvents == null) ? 0 : currentEvents.size();
                        String actionResultJson = invokeAgentMethod("act", activity.getGoal(), history, contextJson);
                        logger.info("🛠️ Action Result: {}", actionResultJson);
                        String toolName = null;
                        try {
                            JsonNode node = objectMapper.readTree(actionResultJson);
                            if (node.has("tool_name") && !node.get("tool_name").isNull()) {
                                toolName = node.get("tool_name").asText();
                            }
                        } catch (Exception e) {
                            logger.warn("⚠️ Invalid JSON in ACT response: {}", actionResultJson);
                        }

                        Map<String, Object> snapshot = snapshotVariables(activityUuid);
                        activity.addStep(new ReasoningStep("act", activity.getGoal(), actionResultJson, snapshot));

                        if (toolName != null && !toolName.isEmpty() && !toolName.equalsIgnoreCase("null")) {

                            logger.info("🛠️ Tool Call Detected: '{}'. Checking for immediate events...", toolName);

                            // Check if events arrived WHILE the LLM was thinking/acting
                            List<JsonNode> updatedEvents = eventsPerActivity.get(activityUuid);
                            int newEventCount = (updatedEvents == null) ? 0 : updatedEvents.size();

                            if (newEventCount > initialEventCount) {

                                // Race condition handled: event arrived while we were busy.
                                logger.info("⚡ Event arrived DURING action execution! Skipping suspension for Activity {}.", activityUuid);

                                // Do not suspend — go directly to OBSERVATION to process the already-arrived event
                                activity.setStatus(Activity.Status.OBSERVATION);
                                activityQueue.offer(activity);

                            } else {
                                // Normal case: no new event, suspend and wait
                                logger.info("💤 Suspending Activity {} (Waiting for future event)", activityUuid);
                                activity.setStatus(Activity.Status.WAITING_FOR_EVENT);
                                pendingActivities.put(activityUuid, activity);
                            }

                        } else {

                            logger.info("⏩ No Tool Call. Proceeding to OBSERVE immediately.");

                            activity.setStatus(Activity.Status.OBSERVATION);
                            activityQueue.offer(activity);
                        }
                    }
                    case OBSERVATION -> {
                        // 1. Serialize the Java list to JSON string for the LLM
                        List<JsonNode> eventsList = eventsPerActivity.getOrDefault(activityUuid, Collections.emptyList());
                        String eventsJson = "[]";
                        try {
                            eventsJson = objectMapper.writeValueAsString(eventsList);
                        } catch (Exception e) {
                            logger.warn("Failed to serialize events", e);
                        }
                        logger.debug("Serialized events for activity {}: {}", activityUuid, eventsJson);

                        // 2. Invoke: pass 'eventsJson' (String) instead of the list
                        String obsResult = invokeAgentMethod("observe", activity.getGoal(), history, contextJson, eventsJson);

                        Map<String, Object> snapshot = snapshotVariables(activityUuid);
                        activity.addStep(new ReasoningStep("observe", activity.getGoal(), obsResult, snapshot));

                        // 3. Consume: clear read events to avoid reprocessing on next loop
                        if (eventsPerActivity.containsKey(activityUuid)) {
                            eventsPerActivity.get(activityUuid).clear();
                            logger.debug("🧹 Cleared consumed events for activity {}", activityUuid);
                        }

                        // Standard completion logic
                        boolean completed = parseCompleted(obsResult);
                        if (completed) {
                            activity.setStatus(Activity.Status.COMPLETED);
                            logger.info("Activity {} marked COMPLETED by observe", activity.getUuid());
                        } else {
                            activity.setStatus(Activity.Status.REASONING);
                            logger.info("Activity {} cycled back to REASONING", activity.getUuid());
                        }
                        activityQueue.offer(activity);
                    }
                    case COMPLETED -> {
                        logger.debug("Activity {} already completed", activity.getUuid());
                    }
                    default -> {
                        logger.warn("Unknown activity status for {}: {}", activity.getUuid(), status);
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.info("Event loop interrupted");
            } catch (Exception e) {
                logger.error("Error while processing activity", e);
            }
        }
        logger.info("🛑 Agent event loop stopped");
    }

    private boolean parseCompleted(String obsResult) {
        if (obsResult == null) return false;
        // try parse JSON: { "completed": true } or nested
        try {
            JsonNode node = objectMapper.readTree(obsResult);
            if (node.has("completed") && node.get("completed").isBoolean()) {
                return node.get("completed").asBoolean();
            }
            // also accept {"result": {"completed": true}} style
            if (node.has("result") && node.get("result").has("completed")) {
                return node.get("result").get("completed").asBoolean(false);
            }
        } catch (Exception ignored) {
            // not JSON - fall back to text match
        }
        String s = obsResult.trim().toLowerCase();
        return s.contains("completed") || s.equals("done") || s.equals("success") || s.equals("goal achieved");
    }

    // Snapshot variables converting JsonNode values to Object for a specific activity UUID
    private Map<String, Object> snapshotVariables(String activityUuid) {
        Map<String, Object> snapshot = new HashMap<>();
        if (activityUuid == null) return snapshot;
        Map<String, JsonNode> vars = variablesPerActivity.get(activityUuid);
        if (vars == null) return snapshot;
        vars.forEach((k, v) -> snapshot.put(k, v));
        return snapshot;
    }

    // Reflectively invoke a method on the agentBrain if available: reason / act / observe
    private String invokeAgentMethod(String methodName, Object... args) {
        try {
            Method target = null;
            for (Method m : agentBrain.getClass().getMethods()) {
                if (m.getName().equalsIgnoreCase(methodName) && m.getParameterCount() == args.length) {
                    target = m;
                    break;
                }
            }
            if (target == null) {
                for (Method m : agentBrain.getClass().getMethods()) {
                    if (m.getName().equalsIgnoreCase(methodName)) {
                        target = m;
                        break;
                    }
                }
            }
            if (target == null) {
                logger.debug("Agent brain has no method '{}'", methodName);
                return "";
            }

            Object result;
            int paramCount = target.getParameterCount();
            if (paramCount == 0) {
                result = target.invoke(agentBrain);
            } else {
                Object[] invokeArgs = args;
                if (args.length != paramCount) {
                    invokeArgs = new Object[paramCount];
                    for (int i = 0; i < Math.min(args.length, paramCount); i++) {
                        invokeArgs[i] = args[i];
                    }
                    for (int i = args.length; i < paramCount; i++) {
                        invokeArgs[i] = null;
                    }
                }
                result = target.invoke(agentBrain, invokeArgs);
            }
            return result == null ? "" : result.toString();
        } catch (Exception e) {
            logger.error("Failed to invoke agent method '{}'", methodName, e);
            return "";
        }
    }

    public void shutdown() {
        loopRunning.set(false);
        executor.shutdownNow();
    }

    private void startSseListener() {
        logger.info("🎧 Starting Async SSE Listener on {}", sseUrl);
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(sseUrl))
                .header("Accept", "text/event-stream")
                .build();

        client.sendAsync(request, HttpResponse.BodyHandlers.fromLineSubscriber(new Flow.Subscriber<String>() {
            private Flow.Subscription sub;
            @Override public void onSubscribe(Flow.Subscription sub) { this.sub = sub; sub.request(1); }
            @Override public void onNext(String line) {
                if (line.startsWith("data:")) {
                    String json = line.substring(5).trim();
                    if (!json.isEmpty() && !json.equals("[DONE]") && json.contains("{")) {
                        logger.debug("📡 SSE Event: {}", json);
                        handleMcpEvent(json);
                    }
                }
                sub.request(1);
            }
            @Override public void onError(Throwable t) { logger.error("❌ SSE Error", t); }
            @Override public void onComplete() { logger.info("✅ SSE Stream Closed"); }
        }));
    }

    private void handleMcpEvent(String json) {
        try {
            JsonNode root = objectMapper.readTree(json);
            JsonNode params = root.has("params") ? root.get("params") : root;

            String mcpType = null;
            if (params.has("mcpType")) {
                mcpType = params.get("mcpType").asText();
            }
            String msgUuid = params.has("uuid") ? params.get("uuid").asText() : "global";

            if ("variable".equalsIgnoreCase(mcpType)) {
                // support both "name" and "key" as the variable identifier
                String name = null;
                if (params.has("name")) name = params.get("name").asText();
                else if (params.has("key")) name = params.get("key").asText();
                else name = "var_" + UUID.randomUUID();

                JsonNode value = params.has("value") ? params.get("value") : NullNode.getInstance();

                variablesPerActivity.computeIfAbsent(msgUuid, k -> new ConcurrentHashMap<>()).put(name, value);

                logger.info("🔁 Variable stored/updated for uuid {}: {} -> {}", msgUuid, name, value);

                return;
            }

            if ("event".equalsIgnoreCase(mcpType)) {
                JsonNode eventPayload = null;
                if (params.has("payload")) eventPayload = params.get("payload");
                else if (params.has("event")) eventPayload = params.get("event");
                else eventPayload = params;

                if (eventPayload.has("event") && eventPayload.get("event").isObject() && !eventPayload.has("key")) {
                    eventPayload = eventPayload.get("event");
                }

                if (msgUuid != null) {
                    // 1. Store the event (memory)
                    eventsPerActivity.computeIfAbsent(msgUuid, k -> new CopyOnWriteArrayList<>()).add(eventPayload);
                    logger.info("📥 Event stored for uuid {}: {}", msgUuid, eventPayload);

                    // 2. Resume: wake up any activity waiting for this UUID
                    Activity pending = pendingActivities.remove(msgUuid); // remove from pending map
                    if (pending != null) {
                        pending.setStatus(Activity.Status.OBSERVATION); // set next state
                        activityQueue.offer(pending); // put back into the main loop
                        logger.info("🔔 WAKING UP Activity {} -> Resumed to OBSERVATION", msgUuid);
                    } else {
                        // If the activity is not pending (maybe finished or running), just log
                        logger.debug("Event received for {} but activity is not in PENDING state.", msgUuid);
                    }
                } else {
                    logger.warn("Received event without UUID, cannot route to activity: {}", eventPayload);
                }
                return;
            }

            logger.info("⚪ Ignored MCP message (not event/variable): {}", params);

        } catch (Exception e) {
            logger.error("Failed to handle MCP event", e);
        }
    }

    private String extractActivityHistory(Activity activity, int windowSize) {
        StringBuilder historyBuilder = new StringBuilder();
        List<ReasoningStep> steps = activity.getHistory();
        int start = Math.max(0, steps.size() - windowSize);
        for (int i = start; i < steps.size(); i++) {
            ReasoningStep step = steps.get(i);
            historyBuilder.append(step.toJson()).append("\n");
        }
        return historyBuilder.toString();
    }

    public T brain() {
        return agentBrain;
    }

    public static class Builder<T extends ReactBrain> {
        private OpenAiChatModel model;
        private Class<T> agentInterface;
        private Object[] tools;
        private ArrayList<Document> documents;
        private McpToolProvider mcpToolProvider;
        private String sseUrl;

        public Builder<T> model(OpenAiChatModel model) {
            this.model = model;
            return this;
        }

        public Builder<T> agentInterface(Class<T> agentInterface) {
            this.agentInterface = agentInterface;
            return this;
        }

        public Builder<T> mcpToolProvider(McpToolProvider provider) {
            this.mcpToolProvider = provider;
            return this;
        }

        public Builder<T> sseUrl(String url) {
            this.sseUrl = url;
            return this;
        }

        public Builder<T> tools(Object... tools) {
            this.tools = tools;
            return this;
        }

        public Builder<T> documents(ArrayList<Document> documents) {
            this.documents = documents;
            return this;
        }

        public AsyncAgent<T> build() {
            Objects.requireNonNull(model, "model must not be null");
            Objects.requireNonNull(agentInterface, "agentInterface must not be null");
            return new AsyncAgent<>(this);
        }
    }
}

