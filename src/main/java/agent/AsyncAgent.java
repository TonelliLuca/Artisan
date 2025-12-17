package agent;

import agent.activity.Activity;
import agent.activity.ReasoningStep;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.NullNode;
import com.fasterxml.jackson.databind.node.TextNode;
import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class AsyncAgent<T extends ReactBrain> {
    private final ChatModel model;
    private final Class<T> agentInterface;
    private final T agentBrain;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AllMiniLmL6V2EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    private final UUID agentId = UUID.randomUUID();
    private final Logger logger = LoggerFactory.getLogger(AsyncAgent.class);

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final Map<String, Activity> activityRegistry = new ConcurrentHashMap<>();

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

        if (this.sseUrl != null && !this.sseUrl.isEmpty()) {
            startSseListener();
        }

        executor.submit(this::eventLoop);
    }

    public void request(String request) {
        if (request == null || request.isBlank()) return;
        Activity activity = new Activity(request);
        activityRegistry.put(activity.getUuid(), activity);
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
                    activityRegistry.remove(activity.getUuid());
                    logger.debug("Skipping completed activity {}", activity.getUuid());
                    continue;
                }
                Activity.Status status = activity.getStatus();
                String phase = status == null ? "UNKNOWN" : status.name();
                logger.debug("Processing activity {} phase={}", activity.getUuid(), phase);

                String history = this.extractActivityHistory(activity, WINDOW_SIZE);
                String activityUuid = activity.getUuid();
                String contextJson;
                try {
                    Map<String, Object> ctx = new HashMap<>();
                    ctx.put("activityUuid", activityUuid);
                    ctx.put("variables", activity.getBeliefsSnapshot());
                    contextJson = objectMapper.writeValueAsString(ctx);
                } catch (Exception e) {
                    logger.debug("Failed to serialize context", e);
                    contextJson = "";
                }

                // ---  PROGRESS TRACKER EXTRACTION ---
                String progressTracker = "(No plan yet. Create one in Observation phase.)";
                JsonNode progressNode = activity.getBelief("goal_progress");
                if (progressNode != null && !progressNode.isNull()) {
                    progressTracker = progressNode.asText();
                }

                switch (status) {
                    case REASONING -> {
                        if (activity.hasEvents()) {
                            logger.info("⚡ Events pending for Activity {} in REASONING phase. Skipping to OBSERVATION.", activityUuid);
                            activity.setStatus(Activity.Status.OBSERVATION);
                            activityQueue.offer(activity);
                            break;
                        }
                        String reasoningResult = invokeAgentMethod("reason", activity.getGoal(), history, contextJson, progressTracker);

                        Map<String, Object> snapshot = activity.getBeliefsSnapshot();
                        activity.addStep(new ReasoningStep("reason", activity.getGoal(), reasoningResult, snapshot));

                        activity.setStatus(Activity.Status.ACTION);
                        logger.info("Activity {} moved to ACTION", activity.getUuid());
                        activityQueue.offer(activity);
                    }
                    case ACTION -> {

                        String actionResultJson = invokeAgentMethod("act", activity.getGoal(), history, contextJson, progressTracker);
                        logger.info("🛠️ Action Result: {}", actionResultJson);

                        String toolName = null;
                        try {
                            String cleanAction = cleanJson(actionResultJson);
                            JsonNode node = objectMapper.readTree(cleanAction);
                            if (node.has("tool_name") && !node.get("tool_name").isNull()) {
                                toolName = node.get("tool_name").asText();
                            }
                        } catch (Exception e) {
                            logger.warn("⚠️ Invalid JSON in ACT response: {}", actionResultJson);
                        }

                        Map<String, Object> snapshot = activity.getBeliefsSnapshot();
                        activity.addStep(new ReasoningStep("act", activity.getGoal(), actionResultJson, snapshot));

                        if (toolName != null && !toolName.isEmpty() && !toolName.equalsIgnoreCase("null")) {
                            logger.info("🛠️ Tool Call Detected: '{}'. Checking for immediate events...", toolName);
                            if (activity.hasEvents()) {
                                logger.info("⚡ Event arrived DURING action execution! Skipping suspension for Activity {}.", activityUuid);
                                activity.setStatus(Activity.Status.OBSERVATION);
                                activityQueue.offer(activity);
                            } else {
                                logger.info("💤 Suspending Activity {} (Waiting for future event)", activityUuid);
                                activity.setStatus(Activity.Status.WAITING_FOR_EVENT);
                            }
                        } else {
                            logger.info("⏩ No Tool Call. Proceeding to OBSERVE immediately.");
                            activity.setStatus(Activity.Status.OBSERVATION);
                            activityQueue.offer(activity);
                        }
                    }
                    case OBSERVATION -> {
                        List<JsonNode> eventsList = activity.consumeEvents();
                        String eventsJson = "[]";
                        try {
                            eventsJson = objectMapper.writeValueAsString(eventsList);
                        } catch (Exception e) {
                            logger.warn("Failed to serialize events", e);
                        }
                        logger.debug("Serialized events for activity {}: {}", activityUuid, eventsJson);

                        String obsResult = invokeAgentMethod("observe", activity.getGoal(), history, contextJson, eventsJson, progressTracker);

                        // --- 2. UPDATE PROGRESS & VARIABLES ---
                        try {
                            String cleanObs = cleanJson(obsResult);
                            JsonNode obsNode = objectMapper.readTree(cleanObs);

                            // A. Update PROGRESS TRACKER
                            if (obsNode.has("new_progress")) {
                                String newProgress = obsNode.get("new_progress").asText();
                                // Save as special TextNode variable
                                activity.setBelief("goal_progress", TextNode.valueOf(newProgress));
                                logger.info("📝 PROGRESS UPDATED for {}:\n{}", activityUuid, newProgress);
                            }

                            // B. Update other variables (technical beliefs)
                            if (obsNode.has("update_variables") && obsNode.get("update_variables").isObject()) {
                                JsonNode updates = obsNode.get("update_variables");
                                updates.fields().forEachRemaining(entry -> {
                                    activity.setBelief(entry.getKey(), entry.getValue());
                                    logger.info("🧠 Belief Update for {}: {} -> {}", activityUuid, entry.getKey(), entry.getValue());
                                });
                            }
                        } catch (Exception e) { /* ignore non-json */ }

                        Map<String, Object> snapshot = activity.getBeliefsSnapshot();
                        activity.addStep(new ReasoningStep("observe", activity.getGoal(), obsResult, snapshot, eventsList));



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
        if (obsResult == null || obsResult.isBlank()) return false;

        try {
            String cleanObs = cleanJson(obsResult);
            JsonNode node = objectMapper.readTree(cleanObs);

            if (node.has("completed")) {
                JsonNode completedNode = node.get("completed");
                if (completedNode.isBoolean()) {
                    return completedNode.asBoolean();
                }
                if (completedNode.isTextual()) {
                    return Boolean.parseBoolean(completedNode.asText());
                }
            }

            if (node.has("result")) {
                JsonNode resultNode = node.get("result");
                if (resultNode.has("completed")) {
                    return resultNode.get("completed").asBoolean(false);
                }
            }

        } catch (Exception e) {

            logger.warn("⚠️ Could not parse JSON for completion check. Keeping activity alive. Response: {}", obsResult);
        }

        return false;
    }

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
            if (msgUuid == null) {
                logger.warn("Received MCP message without UUID, ignoring: {}", params);
                return;
            }
            Activity targetActivity = activityRegistry.get(msgUuid);
            if (targetActivity == null) {
                logger.debug("Received message for unknown or completed activity: {}", msgUuid);
                return;
            }
            if ("variable".equalsIgnoreCase(mcpType)) {
                String name = null;
                if (params.has("name")) name = params.get("name").asText();
                else if (params.has("key")) name = params.get("key").asText();
                else name = "var_" + UUID.randomUUID();

                JsonNode value = params.has("value") ? params.get("value") : NullNode.getInstance();

                targetActivity.setBelief(name, value);
                logger.info("🔁 Belief stored in Activity {}: {} -> {}", msgUuid, name, value);
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

                targetActivity.pushEvent(eventPayload);
                logger.info("📥 Event pushed to Activity {}: {}", msgUuid, eventPayload);

                if (targetActivity.getStatus() == Activity.Status.WAITING_FOR_EVENT) {
                    targetActivity.setStatus(Activity.Status.OBSERVATION);
                    activityQueue.offer(targetActivity);
                    logger.info("🔔 WAKING UP Activity {} -> Resumed to OBSERVATION", msgUuid);
                } else {
                    logger.debug("Event received for {} but activity is busy ({}). Event queued inside activity.", msgUuid, targetActivity.getStatus());
                }
                return;
            }

            logger.info("⚪ Ignored MCP message (not event/variable): {}", params);

        } catch (Exception e) {
            logger.error("Failed to handle MCP event", e);
        }
    }

    private String cleanJson(String response) {
        if (response == null) return "{}";
        String trimmed = response.trim();

        // Remove markdown blocks ```json ... ``` or ``` ... ```
        if (trimmed.startsWith("```")) {
            int start = trimmed.indexOf("{");
            int end = trimmed.lastIndexOf("}");
            if (start != -1 && end != -1) {
                return trimmed.substring(start, end + 1);
            }
        }
        return trimmed;
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
        private ChatModel model;
        private Class<T> agentInterface;
        private Object[] tools;
        private ArrayList<Document> documents;
        private McpToolProvider mcpToolProvider;
        private String sseUrl;


        public Builder<T> model(ChatModel model) {
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

