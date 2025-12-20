package agent.memory;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.stream.Collectors;

public class AgentMemory {

    // Embedding model that converts text to numeric vectors
    private final EmbeddingModel embeddingModel;

    // In-memory embedding store for now
    private final EmbeddingStore<TextSegment> embeddingStore;

    public AgentMemory() {
        // Initialize here to keep other classes simple
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        this.embeddingStore = new InMemoryEmbeddingStore<>();
    }

    /**
     * Save a new episodic memory
     */
    public void save(EpisodicMemory memory) {
        // Convert memory to text for embedding
        String textContent = memory.toTextContent();

        // Create a TextSegment with metadata (useful for future filtering)
        Metadata metadata = new Metadata();
        metadata.put("outcome", "SUCCESS"); // Example: store only successes in the vector DB
        metadata.put("original_goal", memory.getOriginalGoal());

        TextSegment segment = TextSegment.from(textContent, metadata);

        // Compute the embedding vector
        Embedding embedding = embeddingModel.embed(segment).content();

        // Add to the store
        embeddingStore.add(embedding, segment);

        System.out.println("Memory saved: " + memory.getOriginalGoal());
    }

    /**
     * Retrieve memories similar to a new goal (for RAG)
     */
    public List<String> retrieveRelevantMemories(String currentGoal, int maxResults) {
        // Compute embedding for the new goal
        Embedding queryEmbedding = embeddingModel.embed(currentGoal).content();

        // Build the search request
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(maxResults)
                .minScore(0.6) // Apply a minimum score filter for efficiency
                .build();

        // Execute the search
        EmbeddingSearchResult<TextSegment> result = embeddingStore.search(request);

        // Return only the text of matched memories
        return result.matches().stream()
                .map(match -> match.embedded().text())
                .collect(Collectors.toList());
    }
}