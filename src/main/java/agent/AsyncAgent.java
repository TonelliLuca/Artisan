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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class AsyncAgent<T extends ReactBrain> {
    private final OpenAiChatModel model;
    private final Class<T> agentInterface;
    private final T agentBrain;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AllMiniLmL6V2EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    private final UUID agentId = UUID.randomUUID();
    private final Logger logger = LoggerFactory.getLogger(AsyncAgent.class);

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final Map<String, JsonNode> variables = new ConcurrentHashMap<>();
    private final List<JsonNode> events = new CopyOnWriteArrayList<>();

    private final String sseUrl;

    private final BlockingQueue<Activity> activityQueue = new LinkedBlockingQueue<>();
    private final AtomicBoolean loopRunning = new AtomicBoolean(true);

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


    // Modified eventLoop() in src/main/java/agent/AsyncAgent.java
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
                String lastStepJson = activity.lastStep().map(ReasoningStep::toJson).orElse("");
                String contextJson;
                try {
                    Map<String, Object> ctx = new HashMap<>();
                    ctx.put("variables", snapshotVariables()); // copy of current variables
                    ctx.put("events", events); // current events list (JsonNode)
                    contextJson = objectMapper.writeValueAsString(ctx);
                } catch (Exception e) {
                    logger.debug("Failed to serialize context, using empty context", e);
                    contextJson = "";
                }

                switch (status) {
                    case REASONING -> {
                        String reasoningResult = invokeAgentMethod("reason", activity.getGoal(), lastStepJson, contextJson);
                        Map<String, Object> snapshot = snapshotVariables();
                        activity.addStep(new ReasoningStep("reason", activity.getGoal(), reasoningResult, snapshot));
                        activity.setStatus(Activity.Status.ACTION);
                        logger.info("Activity {} moved to ACTION", activity.getUuid());
                    }
                    case ACTION -> {
                        // Tools / external calls should only happen in ACTION.
                        String actionResult = invokeAgentMethod("act", activity.getGoal(), lastStepJson, contextJson);
                        Map<String, Object> snapshot = snapshotVariables();
                        activity.addStep(new ReasoningStep("act", activity.getGoal(), actionResult, snapshot));
                        activity.setStatus(Activity.Status.OBSERVATION);
                        logger.info("Activity {} moved to OBSERVATION", activity.getUuid());
                    }
                    case OBSERVATION -> {
                        String obsResult = invokeAgentMethod("observe", activity.getGoal(), lastStepJson, contextJson);
                        Map<String, Object> snapshot = snapshotVariables();
                        activity.addStep(new ReasoningStep("observe", activity.getGoal(), obsResult, snapshot));

                        boolean completed = parseCompleted(obsResult);
                        if (completed) {
                            activity.setStatus(Activity.Status.COMPLETED);
                            logger.info("Activity {} marked COMPLETED by observe", activity.getUuid());
                        } else {
                            // cycle back to reasoning by default
                            activity.setStatus(Activity.Status.REASONING);
                            logger.info("Activity {} cycled back to REASONING", activity.getUuid());
                        }
                    }
                    case COMPLETED -> {
                        logger.debug("Activity {} already completed", activity.getUuid());
                    }
                    default -> {
                        logger.warn("Unknown activity status for {}: {}", activity.getUuid(), status);
                    }
                }
                // requeue for next phase
                activityQueue.offer(activity);
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

    // Snapshot variables converting JsonNode values to Object
    private Map<String, Object> snapshotVariables() {
        Map<String, Object> snapshot = new HashMap<>();
        variables.forEach(snapshot::put);
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

            if ("variable".equalsIgnoreCase(mcpType)) {

                // support both "name" and "key" as the variable identifier
                String name = null;
                if (params.has("name")) name = params.get("name").asText();
                else if (params.has("key")) name = params.get("key").asText();
                else name = "var_" + UUID.randomUUID();

                JsonNode value = params.has("value") ? params.get("value") : NullNode.getInstance();

                variables.put(name, value);
                logger.info("🔁 Variable stored/updated: {} -> {}", name, value);

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

                events.add(eventPayload);
                logger.info("📥 Event stored: {}", eventPayload);

                return;
            }

            logger.info("⚪ Ignored MCP message (not event/variable): {}", params);

        } catch (Exception e) {
            logger.error("Failed to handle MCP event", e);
        }
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