import agent.AsyncAgent;
import agent.ReactBrain;
import agent.activity.Activity;
import agent.activity.ReasoningStep;
import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpOperationHandler;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.net.Socket;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

import static org.junit.jupiter.api.Assertions.*;

public class ActivitySubmissionIntegrationTest {

    // Minimal NoOp handler used in other tests
    static class NoOpHandler extends McpOperationHandler {
        public NoOpHandler(McpTransport t) {
            super(new java.util.concurrent.ConcurrentHashMap<>(), null, t, l -> {}, () -> {});
        }
        @Override public void handle(JsonNode node) { super.handle(node); }
    }

    private void printActivityHistory(Activity activity) {
        System.out.println("Activity " + activity.getUuid() + " history (steps=" + activity.getHistory().size() + "):");
        activity.getHistory().forEach(step -> {
            System.out.printf("%s | %s%n  input: %s%n  result: %s%n  beliefs: %s%n",
                    step.getTimestamp(),
                    step.getAction(),
                    step.getInput(),
                    step.getResult(),
                    step.getBeliefsSnapshot());
        });
        System.out.println("Full JSON: " + activity.toJson());
    }

    @Test
    void submitActivity_setsTimer_and_activityHistoryEvolves_reactiveSubscribe() throws Exception {
        // 0. Skip if local node server not running
        try (Socket s = new Socket("localhost", 3001)) { /* ok */ } catch (Exception e) {
            System.out.println("⚠️ Node Server not running on 3001. Skipping integration test.");
            return;
        }

        // 1. Setup MCP stack (transport -> client -> provider)
        McpTransport transport = new StreamableHttpMcpTransport.Builder()
                .url("http://localhost:3001/mcp")
                .logRequests(true)
                .logResponses(true)
                .build();
        transport.start(new NoOpHandler(transport));

        McpClient client = new DefaultMcpClient.Builder()
                .transport(transport)
                .toolExecutionTimeout(Duration.ofSeconds(10))
                .build();

        McpToolProvider provider = McpToolProvider.builder()
                .mcpClients(List.of(client))
                .build();

        // 2. Setup model and build agent (same approach as other tests)
        OpenAiChatModel model = OpenAiChatModel.builder()
                .baseUrl("http://langchain4j.dev/demo/openai/v1")
                .apiKey("demo")
                .modelName("gpt-4o-mini")
                .build();

        AsyncAgent<ReactBrain> agent = new AsyncAgent.Builder<ReactBrain>()
                .model(model)
                .agentInterface(ReactBrain.class)
                .mcpToolProvider(provider)
                .sseUrl("http://localhost:3001/sse")
                .build();

        // 3. Prepare an Activity that instructs the agent to subscribe first and then set the timer
        String activityGoal = "First subscribe to the notifications system, then use the timer tool to set a timer: set seconds 10 name test-timer";
        Activity activity = new Activity(activityGoal);

        // 4. Insert the Activity into the agent's internal queue (so test keeps the reference)
        Field qField = AsyncAgent.class.getDeclaredField("activityQueue");
        qField.setAccessible(true);
        @SuppressWarnings("unchecked")
        BlockingQueue<Activity> queue = (BlockingQueue<Activity>) qField.get(agent);
        assertNotNull(queue);
        queue.offer(activity);

        // 5. Wait for the activity to be processed through phases (history grows)
        long deadline = System.currentTimeMillis() + 100000;
        boolean historyEvolved = false;
        while (System.currentTimeMillis() < deadline) {
            int size = activity.getHistory().size();
            if (size >= 3) { // at least reason, act
                historyEvolved = true;
                break;
            }
            Thread.sleep(200);
        }
        assertTrue(historyEvolved, "Activity history should have at least 3 steps (reason, act, observe)");

        // 6. Validate the sequence of actions recorded
        List<ReasoningStep> hist = activity.getHistory();
        assertTrue(hist.size() >= 3, "history length");
        assertEquals("reason", hist.get(0).getAction().toLowerCase());
        assertEquals("act", hist.get(1).getAction().toLowerCase());
        assertEquals("observe", hist.get(2).getAction().toLowerCase());

        // 7. Also verify MCP side-effects: variables and events contain timer entries (per-UUID maps)
        Field varsPerActivityField = AsyncAgent.class.getDeclaredField("variablesPerActivity");
        varsPerActivityField.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<String, Map<String, JsonNode>> varsPerActivity = (Map<String, Map<String, JsonNode>>) varsPerActivityField.get(agent);

        Field eventsPerActivityField = AsyncAgent.class.getDeclaredField("eventsPerActivity");
        eventsPerActivityField.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<String, List<JsonNode>> eventsPerActivity = (Map<String, List<JsonNode>>) eventsPerActivityField.get(agent);

        long deadline2 = System.currentTimeMillis() + 30000;
        boolean gotVar = false;
        boolean gotEvent = false;
        String expectedKey = "test-timer";
        String messageUuid = activity.getUuid();

        while (System.currentTimeMillis() < deadline2 && !(gotVar && gotEvent)) {
            // check per-uuid vars specifically for this messageUuid
            Map<String, JsonNode> bucket = varsPerActivity.get(messageUuid);
            if (!gotVar && bucket != null && bucket.containsKey(expectedKey)) {
                gotVar = true;
            }

            // check per-uuid events specifically for this messageUuid
            List<JsonNode> evList = eventsPerActivity.get(messageUuid);
            if (!gotEvent && evList != null) {
                for (JsonNode ev : evList) {
                    if (ev != null && ev.has("key") && expectedKey.equals(ev.get("key").asText())
                            && ev.has("name") && "timer.finished".equals(ev.get("name").asText())) {
                        gotEvent = true;
                        break;
                    }
                }
            }

            // fallback: scan all per-uuid buckets if not found under the activity UUID
            if (!gotVar) {
                for (Map<String, JsonNode> b : varsPerActivity.values()) {
                    if (b != null && b.containsKey(expectedKey)) {
                        gotVar = true;
                        break;
                    }
                }
            }
            if (!gotEvent) {
                for (List<JsonNode> list : eventsPerActivity.values()) {
                    if (list == null) continue;
                    for (JsonNode ev : list) {
                        if (ev != null && ev.has("key") && expectedKey.equals(ev.get("key").asText())
                                && ev.has("name") && "timer.finished".equals(ev.get("name").asText())) {
                            gotEvent = true;
                            break;
                        }
                    }
                    if (gotEvent) break;
                }
            }

            if (gotVar && gotEvent) break;
            Thread.sleep(200);
        }

        assertTrue(gotVar, "Variable with key '" + expectedKey + "' should be stored in per-UUID maps");

        // locate storedVar in per-uuid map for this activity's UUID (or any per-UUID bucket as fallback)
        JsonNode storedVar = null;
        Map<String, JsonNode> localBucket = varsPerActivity.get(messageUuid);
        if (localBucket != null) storedVar = localBucket.get(expectedKey);
        if (storedVar == null) {
            for (Map<String, JsonNode> b : varsPerActivity.values()) {
                if (b != null && b.containsKey(expectedKey)) {
                    storedVar = b.get(expectedKey);
                    break;
                }
            }
        }

        assertNotNull(storedVar, "Stored variable must be present in per-UUID maps for key '" + expectedKey + "'");
        assertTrue(storedVar.has("seconds") || storedVar.has("name"));

        printActivityHistory(activity);
    }

    @Test
    void submitTwoActivities_concurrently_verifyIsolationAndCompletion() throws Exception {
        // 0. Pre-check Node server
        try (Socket s = new Socket("localhost", 3001)) { /* ok */ } catch (Exception e) {
            System.out.println("⚠️ Node Server not running on 3001. Skipping test.");
            return;
        }

        // 1. Setup stack (transport -> client -> provider)
        McpTransport transport = new StreamableHttpMcpTransport.Builder()
                .url("http://localhost:3001/mcp")
                .logRequests(true)
                .logResponses(true)
                .build();
        transport.start(new NoOpHandler(transport));

        McpClient client = new DefaultMcpClient.Builder()
                .transport(transport)
                .toolExecutionTimeout(Duration.ofSeconds(10))
                .build();

        McpToolProvider provider = McpToolProvider.builder()
                .mcpClients(List.of(client))
                .build();

        // 2. Setup agent
        OpenAiChatModel model = OpenAiChatModel.builder()
                .baseUrl("http://langchain4j.dev/demo/openai/v1")
                .apiKey("demo")
                .modelName("gpt-4o-mini")
                .build();

        AsyncAgent<ReactBrain> agent = new AsyncAgent.Builder<ReactBrain>()
                .model(model)
                .agentInterface(ReactBrain.class)
                .mcpToolProvider(provider)
                .sseUrl("http://localhost:3001/sse")
                .build();

        // 3. Create the TWO activities
        // Activity A: Long (10 seconds)
        String goalLong = "First subscribe to notifications, then set a timer: set seconds 10 name timer-long";
        Activity activityLong = new Activity(goalLong);

        // Activity B: Short (5 seconds) - This should finish BEFORE the other
        String goalShort = "First subscribe to notifications, then set a timer: set seconds 5 name timer-short";
        Activity activityShort = new Activity(goalShort);

        // 4. Enqueue both activities
        Field qField = AsyncAgent.class.getDeclaredField("activityQueue");
        qField.setAccessible(true);
        @SuppressWarnings("unchecked")
        BlockingQueue<Activity> queue = (BlockingQueue<Activity>) qField.get(agent);

        System.out.println("🚀 Submitting BOTH activities...");
        queue.offer(activityLong);
        queue.offer(activityShort);

        // 5. Polling for completion
        long deadline = System.currentTimeMillis() + 30000; // 30s timeout
        boolean longDone = false;
        boolean shortDone = false;

        while (System.currentTimeMillis() < deadline) {
            if (activityLong.isCompleted()) longDone = true;
            if (activityShort.isCompleted()) shortDone = true;

            if (longDone && shortDone) break;
            Thread.sleep(500);
        }

        // 6. Assertions on completion
        assertTrue(shortDone, "Short activity (5s) should be completed");
        assertTrue(longDone, "Long activity (10s) should be completed");

        System.out.println("✅ Both activities completed!");
        printActivityHistory(activityShort);
        printActivityHistory(activityLong);

        // 7. VERIFY ISOLATION (the scientific check)
        // Retrieve agent's internal maps
        Field varsField = AsyncAgent.class.getDeclaredField("variablesPerActivity");
        varsField.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<String, Map<String, JsonNode>> varsPerActivity = (Map<String, Map<String, JsonNode>>) varsField.get(agent);

        // Memory analysis for SHORT activity
        Map<String, JsonNode> shortVars = varsPerActivity.get(activityShort.getUuid());
        assertNotNull(shortVars, "Short activity must have its own variable bucket");
        assertTrue(shortVars.containsKey("timer-short"), "Short bucket must contain 'timer-short'");
        assertFalse(shortVars.containsKey("timer-long"), "CRITICAL: Short bucket MUST NOT contain 'timer-long' (Isolation Check)");

        // Memory analysis for LONG activity
        Map<String, JsonNode> longVars = varsPerActivity.get(activityLong.getUuid());
        assertNotNull(longVars, "Long activity must have its own variable bucket");
        assertTrue(longVars.containsKey("timer-long"), "Long bucket must contain 'timer-long'");
        assertFalse(longVars.containsKey("timer-short"), "CRITICAL: Long bucket MUST NOT contain 'timer-short' (Isolation Check)");

        System.out.println("✅ Context Isolation Verified: Variables did not leak between activities.");
    }
}