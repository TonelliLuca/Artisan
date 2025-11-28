import agent.AsyncAgent;
import agent.ReactBrain;
import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.agentic.Agent;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpClient;
import dev.langchain4j.mcp.client.transport.McpOperationHandler;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.SystemMessage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.*;

public class AgentTest {

    public interface SimpleAgentInterface extends ReactBrain {
        @SystemMessage("This is a simple test agent interface.")
        String test(String input);
    }

    static class NoOpHandler extends McpOperationHandler {
        public NoOpHandler(McpTransport t) {
            super(new ConcurrentHashMap<>(), null, t, l -> {}, () -> {});
        }
        @Override public void handle(JsonNode node) { super.handle(node); }
    }

    @Nested
    @DisplayName("Builder Tests")
    class BuilderTests {

        @Test
        @DisplayName("Builder should create instance with model")
        void builderWithModel() {
            OpenAiChatModel model = OpenAiChatModel.builder()
                    .apiKey("test-key")
                    .build();

            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.model(model);

            assertNotNull(builder);
        }

        @Test
        @DisplayName("Builder should accept agent interface")
        void builderWithInterface() {
            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.agentInterface(SimpleAgentInterface.class);

            assertNotNull(builder);
        }

        @Test
        @DisplayName("Builder should throw exception when model is null")
        void builderFailsWithoutModel() {
            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.agentInterface(SimpleAgentInterface.class);

            assertThrows(NullPointerException.class, builder::build);
        }

        @Test
        @DisplayName("Builder should throw exception when interface is null")
        void builderFailsWithoutInterface() {
            OpenAiChatModel model = OpenAiChatModel.builder()
                    .apiKey("test-key")
                    .build();

            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.model(model);

            assertThrows(NullPointerException.class, builder::build);
        }

        @Test
        @DisplayName("Builder should accept tools")
        void builderWithTools() {
            Object tool = new Object();

            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.tools(tool);

            assertNotNull(builder);
        }

        @Test
        @DisplayName("Builder should accept multiple tools")
        void builderWithMultipleTools() {
            Object tool1 = new Object();
            Object tool2 = new Object();

            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.tools(tool1, tool2);

            assertNotNull(builder);
        }

        @Test
        @DisplayName("Builder should accept documents")
        void builderWithDocuments() {
            ArrayList<Document> docs = new ArrayList<>();
            docs.add(Document.from("test content"));

            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.documents(docs);

            assertNotNull(builder);
        }

        @Test
        @DisplayName("Builder should accept MCP tools")
        void builderWithMcpTools() {
            McpTransport transport = new StreamableHttpMcpTransport.Builder()
                    .url("http://localhost:3001/mcp")
                    .build();
            transport.start(new NoOpHandler(transport));

            McpClient client = new DefaultMcpClient.Builder()
                    .transport(transport)
                    .build();

            McpToolProvider provider = McpToolProvider.builder()
                    .mcpClients(List.of(client))
                    .build();

            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.mcpToolProvider(provider);

            assertNotNull(builder);
        }

        @Test
        @DisplayName("Builder should accept SSE URL")
        void builderWithSseUrl() {
            AsyncAgent.Builder<SimpleAgentInterface> builder = new AsyncAgent.Builder<>();
            builder.sseUrl("http://localhost:3000/sse");

            assertNotNull(builder);
        }

        @Test
        @DisplayName("Builder should create Agent with required parameters")
        void builderCreatesAgent() {
            OpenAiChatModel model = OpenAiChatModel.builder()
                    .baseUrl("http://langchain4j.dev/demo/openai/v1")
                    .apiKey("demo")
                    .modelName("gpt-4o-mini")
                    .build();

            ArrayList<Document> docs = new ArrayList<>();
            docs.add(Document.from("test manual"));

            AsyncAgent<SimpleAgentInterface> agent = new AsyncAgent.Builder<SimpleAgentInterface>()
                    .model(model)
                    .agentInterface(SimpleAgentInterface.class)
                    .tools(new Object())
                    .documents(docs)
                    .build();

            assertNotNull(agent);
        }

        @Test
        @DisplayName("Builder should create Agent with MCP and SSE parameters using Real Objects")
        void builderCreatesAgentWithMcp() {
            OpenAiChatModel model = OpenAiChatModel.builder()
                    .baseUrl("http://langchain4j.dev/demo/openai/v1")
                    .apiKey("demo")
                    .modelName("gpt-4o-mini")
                    .build();

            McpTransport transport = new StreamableHttpMcpTransport.Builder()
                    .url("http://localhost:3001/mcp")
                    .build();
            transport.start(new NoOpHandler(transport));

            McpClient client = new DefaultMcpClient.Builder()
                    .transport(transport)
                    .build();

            McpToolProvider provider = McpToolProvider.builder()
                    .mcpClients(List.of(client))
                    .build();

            AsyncAgent<SimpleAgentInterface> agent = new AsyncAgent.Builder<SimpleAgentInterface>()
                    .model(model)
                    .agentInterface(SimpleAgentInterface.class)
                    .mcpToolProvider(provider)
                    .sseUrl("http://localhost:3001/sse")
                    .build();

            assertNotNull(agent);
            assertNotNull(agent.brain());
        }
    }


    @Nested
    @DisplayName("Agent Functionality Tests")
    class AgentFunctionalityTests {
        public interface McpTestAgent extends ReactBrain {
            @SystemMessage("You are a helpful assistant with access to external tools via MCP. Use them when requested.")
            String chat(String input);
        }

        @Nested
        @DisplayName("MCP Integration Tests")
        class McpIntegrationTests {

            @Test
            @DisplayName("Should actually call MCP Tool via Agent (Integration)")
            void shouldExecuteRealMcpTool() {
                try {
                    new java.net.Socket("localhost", 3001).close();
                } catch (Exception e) {
                    System.out.println("⚠️ Node Server not running on 3001. Skipping integration test.");
                    return;
                }

                System.out.println("🚀 Node Server detected. Starting MCP Tool Call Test...");

                McpTransport transport = new StreamableHttpMcpTransport.Builder()
                        .url("http://localhost:3001/mcp")
                        .logRequests(true)
                        .logResponses(true)
                        .build();
                transport.start(new NoOpHandler(transport));

                McpClient client = new DefaultMcpClient.Builder()
                        .transport(transport)
                        .toolExecutionTimeout(java.time.Duration.ofSeconds(10))
                        .build();

                McpToolProvider provider = McpToolProvider.builder()
                        .mcpClients(List.of(client))
                        .build();

                OpenAiChatModel model = OpenAiChatModel.builder()
                        .baseUrl("http://langchain4j.dev/demo/openai/v1")
                        .apiKey("demo")
                        .modelName("gpt-4o-mini")
                        .build();

                AsyncAgent<McpTestAgent> agent = new AsyncAgent.Builder<McpTestAgent>()
                        .model(model)
                        .agentInterface(McpTestAgent.class)
                        .mcpToolProvider(provider)
                        .sseUrl("http://localhost:3001/sse")
                        .build();

                System.out.println("🤖 Asking agent to use the tool...");

                String response = agent.brain().chat("Subscribe to the notification system, please.");

                System.out.println("🤖 Agent Response: " + response);
                assertNotNull(response);
                assertFalse(response.isEmpty());
            }
        }

        @Test
        @DisplayName("Should set timer via MCP and store variable and event via SSE (Integration)")
        void shouldSetTimerAndReceiveVariableAndEvent() throws Exception {
            try {
                new java.net.Socket("localhost", 3001).close();
            } catch (Exception e) {
                System.out.println("⚠️ Node Server not running on 3001. Skipping integration test.");
                return;
            }

            System.out.println("🚀 Node Server detected. Starting full MCP set+SSE flow test...");

            McpTransport transport = new StreamableHttpMcpTransport.Builder()
                    .url("http://localhost:3001/mcp")
                    .logRequests(true)
                    .logResponses(true)
                    .build();
            transport.start(new NoOpHandler(transport));

            McpClient client = new DefaultMcpClient.Builder()
                    .transport(transport)
                    .toolExecutionTimeout(java.time.Duration.ofSeconds(10))
                    .build();

            McpToolProvider provider = McpToolProvider.builder()
                    .mcpClients(List.of(client))
                    .build();

            OpenAiChatModel model = OpenAiChatModel.builder()
                    .baseUrl("http://langchain4j.dev/demo/openai/v1")
                    .apiKey("demo")
                    .modelName("gpt-4o-mini")
                    .build();

            AsyncAgent<McpTestAgent> agent = new AsyncAgent.Builder<McpTestAgent>()
                    .model(model)
                    .agentInterface(McpTestAgent.class)
                    .mcpToolProvider(provider)
                    .sseUrl("http://localhost:3001/sse")
                    .build();

            System.out.println("🤖 Asking agent to subscribe to notifications...");
            try {
                agent.brain().chat("Subscribe to the notification system, please.");
            } catch (java.lang.reflect.UndeclaredThrowableException ute) {
                System.out.println("⚠️ agent.brain().chat (subscribe) raised UndeclaredThrowableException (continuing)");
                Throwable cause = ute.getCause();
                if (cause != null) {
                    System.out.println("Cause class: " + cause.getClass().getName() + " - " + cause.getMessage());
                    cause.printStackTrace(System.out);
                } else {
                    ute.printStackTrace(System.out);
                }
            }

            Thread.sleep(200);

            System.out.println("🤖 Asking agent to set timer via tool (set seconds=1, name=test-timer)...");
            String prompt = "Use the timer tool to set a timer: set seconds 1 name test-timer";

            try {
                agent.brain().chat(prompt);
            } catch (java.lang.reflect.UndeclaredThrowableException ute) {
                System.out.println("⚠️ agent.brain().chat (set) raised UndeclaredThrowableException (continuing to validate SSE side-effects)");
                Throwable cause = ute.getCause();
                if (cause != null) {
                    System.out.println("Cause class: " + cause.getClass().getName() + " - " + cause.getMessage());
                    cause.printStackTrace(System.out);
                } else {
                    ute.printStackTrace(System.out);
                }
            }

            String expectedKey = "test-timer";

            java.lang.reflect.Field varsField = AsyncAgent.class.getDeclaredField("variables");
            varsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            java.util.Map<String, JsonNode> vars =
                    (java.util.Map<String, JsonNode>) varsField.get(agent);

            java.lang.reflect.Field eventsField = AsyncAgent.class.getDeclaredField("events");
            eventsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            java.util.List<JsonNode> events =
                    (java.util.List<JsonNode>) eventsField.get(agent);

            long deadline = System.currentTimeMillis() + 8000;
            boolean gotVar = false;
            boolean gotEvent = false;
            while (System.currentTimeMillis() < deadline && !(gotVar && gotEvent)) {
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
            JsonNode matchedEvent = null;
            for (JsonNode ev : events) {
                if (ev != null && ev.has("key") && expectedKey.equals(ev.get("key").asText())
                        && ev.has("name") && "timer.finished".equals(ev.get("name").asText())) {
                    matchedEvent = ev;
                    break;
                }
            }
            assertNotNull(matchedEvent);
            assertEquals("timer.finished", matchedEvent.get("name").asText());
        }


        @Test
        @DisplayName("Agent brain should work and correctly call the model")
        void agentBrainFunctionality() {
            AsyncAgent<SimpleAgentInterface> agent = createTestAgent();
            String response = agent.brain().test("Hello, Agent!");
            System.out.println(response);
            assertNotNull(response);
        }

        @Test
        @DisplayName("Agent should accept request")
        void agentAcceptsRequest() {
            AsyncAgent<SimpleAgentInterface> agent = createTestAgent();

            assertDoesNotThrow(() -> agent.request("test request"));
        }

        private AsyncAgent<SimpleAgentInterface> createTestAgent() {
            OpenAiChatModel model = OpenAiChatModel.builder()
                    .baseUrl("http://langchain4j.dev/demo/openai/v1")
                    .apiKey("demo")
                    .modelName("gpt-4o-mini")
                    .build();
            ArrayList<Document> docs = new ArrayList<>();

            return new AsyncAgent.Builder<SimpleAgentInterface>()
                    .model(model)
                    .agentInterface(SimpleAgentInterface.class)
                    .documents(docs)
                    .build();
        }
    }
}

