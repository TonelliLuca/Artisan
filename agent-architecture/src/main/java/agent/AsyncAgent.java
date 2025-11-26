package agent;


import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AsyncAgent<T>{
    private final OpenAiChatModel model;
    private final Class<T> agentInterface;
    private final T agentBrain;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    //local model for embeddings
    private final AllMiniLmL6V2EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

    private State state;
    private final Logger logger = LoggerFactory.getLogger(AsyncAgent.class);

    private AsyncAgent(Builder<T> builder) {
        this.model = builder.model;
        this.agentInterface = builder.agentInterface;
        this.agentBrain = AgenticServices
                .agentBuilder(agentInterface)
                .chatModel(model)
                .tools(builder.tools)
                .beforeAgentInvocation(request -> logger.debug("[BEFORE AGENT] {}", request))
                .afterAgentInvocation(response -> logger.debug("[AFTER AGENT] {}", response))
                .build();

    }

    public void request(String request) {
        return;
    }

    public T brain() {
        return agentBrain;
    }


    public static class Builder<T> {
        private OpenAiChatModel model;
        private Class<T> agentInterface;
        private Object[] tools;
        private ArrayList<Document> documents;

        public Builder<T> model(OpenAiChatModel model) {
            this.model = model;
            return this;
        }


        public Builder<T> agentInterface(Class<T> agentInterface) {
            this.agentInterface = agentInterface;
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

