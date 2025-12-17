import agent.AsyncAgent;
import agent.ReactBrain;
import agent.activity.Activity;
import agent.activity.ReasoningStep;
import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpOperationHandler;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.net.Socket;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

import static org.junit.jupiter.api.Assertions.*;

public class ActivitySubmissionIntegrationTest {

//    OllamaChatModel model = OllamaChatModel.builder()
//            .baseUrl("http://localhost:11434")
//            .modelName("qwen2.5")
//            .temperature(0.0)
//            .timeout(java.time.Duration.ofMinutes(2))
//            .build();

    OpenAiChatModel model = OpenAiChatModel.builder()
            .baseUrl("http://langchain4j.dev/demo/openai/v1")
            .apiKey("demo")
            .modelName("gpt-4o-mini")
            .build();

    static class NoOpHandler extends McpOperationHandler {
        public NoOpHandler(McpTransport t) {
            super(new java.util.concurrent.ConcurrentHashMap<>(), null, t, l -> {}, () -> {});
        }
        @Override public void handle(JsonNode node) { super.handle(node); }
    }

    // Helper updated with optional windowSize parameter
    private String formatActivityHistory(Activity activity) {
        StringBuilder sb = new StringBuilder();
        sb.append("Activity ").append(activity.getUuid())
                .append(" history (steps=").append(activity.getHistory().size()).append("):\n");
        activity.getHistory().forEach(step -> {
            sb.append(String.format("%s | %s%n  input: %s%n  result: %s%n  beliefs: %s%n",
                    step.getTimestamp(),
                    step.getAction(),
                    step.getInput(),
                    step.getResult(),
                    step.getBeliefsSnapshot()));
        });
        sb.append("Full JSON: ").append(activity.toJson()).append("\n");
        return sb.toString();
    }

    private void printActivityHistory(Activity activity) {
        System.out.println(formatActivityHistory(activity));
    }

    @Test
    void submitActivity_setsTimer_and_activityHistoryEvolves_reactiveSubscribe() throws Exception {
        // 0. Skip if local node server not running
        try (Socket s = new Socket("localhost", 3001)) { /* ok */ } catch (Exception e) {
            System.out.println("⚠️ Node Server not running on 3001. Skipping integration test.");
            return;
        }

        // 1. Setup Stack
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



        AsyncAgent<ReactBrain> agent = new AsyncAgent.Builder<ReactBrain>()
                .model(model)
                .agentInterface(ReactBrain.class)
                .mcpToolProvider(provider)
                .sseUrl("http://localhost:3001/sse")
                .build();

        // 2. Prepare Activity
        String activityGoal = "First subscribe to the notifications system, then use the timer tool to set a timer: set seconds 5 name test-timer";
        Activity activity = new Activity(activityGoal);

        // 3. Manual Injection (Simulate request)
        // Manually register the activity in the private registry so SSE works
        Field registryField = AsyncAgent.class.getDeclaredField("activityRegistry");
        registryField.setAccessible(true);
        ((Map<String, Activity>) registryField.get(agent)).put(activity.getUuid(), activity);

        Field qField = AsyncAgent.class.getDeclaredField("activityQueue");
        qField.setAccessible(true);
        ((BlockingQueue<Activity>) qField.get(agent)).offer(activity);

        // 4. Wait for evolution
        long deadline = System.currentTimeMillis() + 120000;
        boolean historyEvolved = false;
        while (System.currentTimeMillis() < deadline) {
            if (activity.getHistory().size() >= 3) {
                historyEvolved = true;
                break;
            }
            Thread.sleep(500);
        }
        assertTrue(historyEvolved, "Activity history should grow");

        // 5. Verify Beliefs (Instead of reflecting on removed Maps)
        // The variable should appear DIRECTLY inside the Activity
        String expectedKey = "test-timer";
        boolean gotVar = false;
        long deadline2 = System.currentTimeMillis() + 30000;

        while (System.currentTimeMillis() < deadline2) {
            JsonNode belief = activity.getBelief(expectedKey);
            if (belief != null) {
                gotVar = true;
                break;
            }
            Thread.sleep(500);
        }

        assertTrue(gotVar, "Variable 'test-timer' should be stored in Activity beliefs via SSE");
        printActivityHistory(activity);
    }

    @Test
    void submitTwoActivities_concurrently_verifyIsolationAndCompletion() throws Exception {
        // Pre-check Node server
        try (Socket s = new Socket("localhost", 3001)) { /* ok */ } catch (Exception e) { return; }

        // Setup usual stack...
        McpTransport transport = new StreamableHttpMcpTransport.Builder()
                .url("http://localhost:3001/mcp").logRequests(false).logResponses(false).build();
        transport.start(new NoOpHandler(transport));
        McpClient client = new DefaultMcpClient.Builder().transport(transport).build();
        McpToolProvider provider = McpToolProvider.builder().mcpClients(List.of(client)).build();


        AsyncAgent<ReactBrain> agent = new AsyncAgent.Builder<ReactBrain>()
                .model(model)
                .agentInterface(ReactBrain.class)
                .mcpToolProvider(provider)
                .sseUrl("http://localhost:3001/sse")
                .build();

        // Two Activities
        Activity activityLong = new Activity("First subscribe, then set timer: 8 seconds name timer-long");
        Activity activityShort = new Activity("First subscribe, then set timer: 3 seconds name timer-short");

        // Manual Injection into Registry & Queue
        Field registryField = AsyncAgent.class.getDeclaredField("activityRegistry");
        registryField.setAccessible(true);
        Map<String, Activity> registry = (Map<String, Activity>) registryField.get(agent);

        registry.put(activityLong.getUuid(), activityLong);
        registry.put(activityShort.getUuid(), activityShort);

        Field qField = AsyncAgent.class.getDeclaredField("activityQueue");
        qField.setAccessible(true);
        BlockingQueue<Activity> queue = (BlockingQueue<Activity>) qField.get(agent);

        System.out.println("🚀 Submitting Concurrent Activities...");
        queue.offer(activityLong);
        queue.offer(activityShort);

        // Polling
        long deadline = System.currentTimeMillis() + 120000;
        while (System.currentTimeMillis() < deadline) {
            if (activityLong.isCompleted() && activityShort.isCompleted()) break;
            Thread.sleep(500);
        }

        assertTrue(activityShort.isCompleted(), "Short activity should finish");
        assertTrue(activityLong.isCompleted(), "Long activity should finish");

        // --- ISOLATION CHECK (New logic) ---
        // Check the Activity objects' memory directly

        // Short must not have Long
        assertNotNull(activityShort.getBelief("timer-short"));
        assertNull(activityShort.getBelief("timer-long"), "CRITICAL: Isolation breach in Short Activity");

        // Long must not have Short
        assertNotNull(activityLong.getBelief("timer-long"));
        assertNull(activityLong.getBelief("timer-short"), "CRITICAL: Isolation breach in Long Activity");

        System.out.println("✅ Context Isolation Verified using Activity Encapsulation.");
    }

    @Test
    void submitActivities_complexScenario() throws Exception {
        // Pre-check
        try (Socket s = new Socket("localhost", 3001)) { /* ok */ } catch (Exception e) { return; }

        // ... (Stack setup identical to others) ...
        McpTransport transport = new StreamableHttpMcpTransport.Builder().url("http://localhost:3001/mcp").build();
        transport.start(new NoOpHandler(transport));
        McpClient client = new DefaultMcpClient.Builder().transport(transport).build();
        McpToolProvider provider = McpToolProvider.builder().mcpClients(List.of(client)).build();

        AsyncAgent<ReactBrain> agent = new AsyncAgent.Builder<ReactBrain>()
                .model(model).agentInterface(ReactBrain.class)
                .mcpToolProvider(provider).sseUrl("http://localhost:3001/sse").build();

        // 1. Define Activities
        Activity actDouble = new Activity("First subscribe. Then set timer 3s 'timer-A'. ALSO set timer 5s 'timer-B'.");
        Activity actImpossible = new Activity("Check Apple stock price.");

        // 2. Inject
        Field registryField = AsyncAgent.class.getDeclaredField("activityRegistry");
        registryField.setAccessible(true);
        ((Map<String, Activity>) registryField.get(agent)).put(actDouble.getUuid(), actDouble);
        ((Map<String, Activity>) registryField.get(agent)).put(actImpossible.getUuid(), actImpossible);

        Field qField = AsyncAgent.class.getDeclaredField("activityQueue");
        qField.setAccessible(true);
        BlockingQueue<Activity> queue = (BlockingQueue<Activity>) qField.get(agent);

        queue.offer(actDouble);
        queue.offer(actImpossible);

        // 3. Wait
        long deadline = System.currentTimeMillis() + 120000;
        while (System.currentTimeMillis() < deadline) {
            if (actDouble.isCompleted() && actImpossible.isCompleted()) break;
            Thread.sleep(500);
        }

        assertTrue(actDouble.isCompleted());
        assertTrue(actImpossible.isCompleted());

        // 4. Verify Double Timer Logic (Using Activity Memory)
        assertNotNull(actDouble.getBelief("timer-A"), "Missing timer-A");
        assertNotNull(actDouble.getBelief("timer-B"), "Missing timer-B");

        // 5. Verify Impossible (Using Result)
        ReasoningStep last = actImpossible.lastStep().orElseThrow();
        // The agent should have concluded it cannot do it
        System.out.println("Impossible Task Result: " + last.getResult());

        System.out.println("✅ Complex Scenario Passed");
        System.out.println("---- Activity Histories ----");
        System.out.println("Double Timer Activity:");
        printActivityHistory(actDouble);
        System.out.println("Impossible Task Activity:");
        printActivityHistory(actImpossible);
    }

    @Test
    void submitSequentialDependentTimers_verifyChecklistLogic() throws Exception {
        // 0. Pre-check
        try (Socket s = new Socket("localhost", 3001)) { /* ok */ } catch (Exception e) { return; }

        // 1. Setup Stack
        McpTransport transport = new StreamableHttpMcpTransport.Builder().url("http://localhost:3001/mcp").build();
        transport.start(new NoOpHandler(transport));
        McpClient client = new DefaultMcpClient.Builder().transport(transport).toolExecutionTimeout(Duration.ofSeconds(10)).build();
        McpToolProvider provider = McpToolProvider.builder().mcpClients(List.of(client)).build();

        // Use your local model


        AsyncAgent<ReactBrain> agent = new AsyncAgent.Builder<ReactBrain>()
                .model(model).agentInterface(ReactBrain.class)
                .mcpToolProvider(provider).sseUrl("http://localhost:3001/sse").build();

        // 2. Goal
        String goal = "First subscribe. Then set 'timer-A' for 4 seconds and 'timer-B' for 4 seconds. " +
                "Wait for BOTH 'timer-A' and 'timer-B' to finish ringing. " +
                "ONLY AFTER both A and B events are received, set 'timer-C' for 2 seconds.";
        Activity activity = new Activity(goal);

        // 3. Inject
        Field registryField = AsyncAgent.class.getDeclaredField("activityRegistry");
        registryField.setAccessible(true);
        ((Map<String, Activity>) registryField.get(agent)).put(activity.getUuid(), activity);

        Field qField = AsyncAgent.class.getDeclaredField("activityQueue");
        qField.setAccessible(true);
        ((BlockingQueue<Activity>) qField.get(agent)).offer(activity);

        // 4. Wait
        System.out.println("⏳ Waiting for sequential timers (approx 15s)...");
        long deadline = System.currentTimeMillis() + 120000;
        while (System.currentTimeMillis() < deadline) {
            if (activity.isCompleted()) break;
            Thread.sleep(1000);
        }
        assertTrue(activity.isCompleted(), "Activity should complete within timeout");

        // 5. Verify Timeline (EVENT-BASED LOGIC)
        List<ReasoningStep> history = activity.getHistory();

        long timeA_Finished = -1;
        long timeB_Finished = -1;
        long timeC_Started = -1;

        System.out.println("\n--- Event Sequence Analysis ---");

        for (ReasoningStep step : history) {
            // Replace the loop logic with this:

            String act = step.getAction();
            String rawResult = step.getResult();
            long timestamp = step.getTimestamp().toEpochMilli();

// A. Detect when events ARRIVE (OBSERVE phase)
            if ("observe".equals(act)) {
                // FIX: Use step.getEvents() instead of step.getInput()
                // Convert the list of JSON nodes to a string for quick checks
                String eventsContent = step.getEvents().toString().toLowerCase();

                if (eventsContent.contains("timer-a") && eventsContent.contains("finished")) {
                    timeA_Finished = timestamp;
                    System.out.println("✅ Event Received: Timer A Finished at " + step.getTimestamp());
                }
                if (eventsContent.contains("timer-b") && eventsContent.contains("finished")) {
                    timeB_Finished = timestamp;
                    System.out.println("✅ Event Received: Timer B Finished at " + step.getTimestamp());
                }
            }

            // B. Detect when the action STARTS (ACT phase)
            if ("act".equals(act)) {
                // Extract only the JSON to ignore future thoughts ("I will do C later...")
                String cleanJson = extractJson(rawResult);

                // If the JSON contains the real tool call
                if (cleanJson.contains("timer-c")) {
                    timeC_Started = timestamp;
                    System.out.println("🚀 Action Executed: Timer C Started at " + step.getTimestamp());
                }
            }
        }

        // 6. Logical assertions
        // CORE OF THE TEST: C must start AFTER A and B have finished
        assertTrue(timeA_Finished > 0, "Timer A finished event missing");
        assertTrue(timeB_Finished > 0, "Timer B finished event missing");
        assertTrue(timeC_Started > 0, "Timer C action missing");

        // IL CUORE DEL TEST: C deve essere partito DOPO che A e B sono finiti
        if (timeC_Started < timeA_Finished || timeC_Started < timeB_Finished) {
            fail("❌ Sequential Violation: Timer C started before A/B finished!\n" +
                    "Time A: " + timeA_Finished + "\n" +
                    "Time B: " + timeB_Finished + "\n" +
                    "Time C: " + timeC_Started);
        }

        System.out.println("✅ Sequential Dependency Logic Verified: C started (" + timeC_Started +
                ") strictly after A (" + timeA_Finished + ") and B (" + timeB_Finished + ")");

        printActivityHistory(activity);
    }

    // Helper to clean the response and get only the action JSON
    private String extractJson(String text) {
        if (text == null) return "";
        int start = text.indexOf("```json");
        if (start == -1) start = text.indexOf("{"); // Fallback se manca markdown
        int end = text.lastIndexOf("}");

        if (start != -1 && end != -1 && end > start) {
            return text.substring(start, end + 1).toLowerCase();
        }
        return "";
    }
}