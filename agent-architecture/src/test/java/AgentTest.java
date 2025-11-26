import agent.AsyncAgent;
import dev.langchain4j.agentic.Agent;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.SystemMessage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

public class AgentTest {

    public interface SimpleAgentInterface {
        @Agent("A simple agent interface for testing purposes.")

        @SystemMessage("This is a simple test agent interface.")
        String test(String input);
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
    }


    @Nested
    @DisplayName("Agent Functionality Tests")
    class AgentFunctionalityTests {

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