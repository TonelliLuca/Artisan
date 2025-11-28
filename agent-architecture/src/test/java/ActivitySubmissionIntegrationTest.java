

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
        try (Socket s = new Socket("localhost", 3001)) { /* ok */ }
        catch (Exception e) {
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
        String activityGoal = "First subscribe to the notifications system, then use the timer tool to set a timer: set seconds 1 name test-timer";
        Activity activity = new Activity(activityGoal);

        // 4. Insert the Activity into the agent's internal queue (so test keeps the reference)
        Field qField = AsyncAgent.class.getDeclaredField("activityQueue");
        qField.setAccessible(true);
        @SuppressWarnings("unchecked")
        BlockingQueue<Activity> queue = (BlockingQueue<Activity>) qField.get(agent);
        assertNotNull(queue);
        queue.offer(activity);

        // 5. Wait for the activity to be processed through phases (history grows)
        long deadline = System.currentTimeMillis() + 15000;
        boolean historyEvolved = false;
        while (System.currentTimeMillis() < deadline) {
            int size = activity.getHistory().size();
            if (size >= 3) { // at least reason, act, observe
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

        // 7. Also verify MCP side-effects: variables and events contain timer entries
        Field varsField = AsyncAgent.class.getDeclaredField("variables");
        varsField.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<String, JsonNode> vars = (Map<String, JsonNode>) varsField.get(agent);

        Field eventsField = AsyncAgent.class.getDeclaredField("events");
        eventsField.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<JsonNode> events = (List<JsonNode>) eventsField.get(agent);

        // wait a bit more for variable + event (timer finished occurs after ~1s)
        long deadline2 = System.currentTimeMillis() + 12000;
        boolean gotVar = false;
        boolean gotEvent = false;
        String expectedKey = "test-timer";
        while (System.currentTimeMillis() < deadline2 && !(gotVar && gotEvent)) {
            if (!gotVar && vars.containsKey(expectedKey)) gotVar = true;
            if (!gotEvent) {
                for (JsonNode ev : events) {
                    if (ev != null && ev.has("key") && expectedKey.equals(ev.get("key").asText())
                            && ev.has("name") && "timer.finished".equals(ev.get("name").asText())) {
                        gotEvent = true;
                        break;
                    }
                }
            }
            if (gotVar && gotEvent) break;
            Thread.sleep(200);
        }

        assertTrue(gotVar, "Variable with key '" + expectedKey + "' should be stored in variables map");
        assertTrue(vars.containsKey(expectedKey));
        JsonNode storedVar = vars.get(expectedKey);
        assertNotNull(storedVar);
        assertTrue(storedVar.has("seconds") || storedVar.has("name"));

        assertTrue(gotEvent, "Events should contain a 'timer.finished' for key '" + expectedKey + "'");
        printActivityHistory(activity);
    }
}