package agent.memory;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

public class EpisodicMemory {
    private final String id;
    private final String originalGoal;
    private final String outcome; // "SUCCESS" or "FAILURE"
    private final String summary;
    private final List<String> successfulProcedure; // Steps that worked
    private final Instant timestamp;

    public EpisodicMemory(String originalGoal, String outcome, String summary, List<String> successfulProcedure) {
        this.id = UUID.randomUUID().toString();
        this.originalGoal = originalGoal;
        this.outcome = outcome;
        this.summary = summary;
        this.successfulProcedure = successfulProcedure;
        this.timestamp = Instant.now();
    }

    // Crucial method for RAG: the text the AI will read when retrieving this memory
    public String toTextContent() {
        return String.format("""
            PAST TASK: %s
            OUTCOME: %s
            SUMMARY: %s
            PROCEDURE USED:
            - %s
            """,
                originalGoal,
                outcome,
                summary,
                String.join("\n- ", successfulProcedure)
        );
    }


    public String getOriginalGoal() { return originalGoal; }

}