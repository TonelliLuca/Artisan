package agent.activity;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class ReasoningStep {
    private final Instant timestamp;
    private final String action;
    private final String input;
    private final String result;
    private final Map<String, Object> beliefsSnapshot;

    public ReasoningStep(String action, String input, String result, Map<String, Object> beliefsSnapshot) {
        this.timestamp = Instant.now();
        this.action = action;
        this.input = input;
        this.result = result;
        this.beliefsSnapshot = beliefsSnapshot == null
                ? Collections.emptyMap()
                : Collections.unmodifiableMap(new HashMap<>(beliefsSnapshot));
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    public String getAction() {
        return action;
    }

    public String getInput() {
        return input;
    }

    public String getResult() {
        return result;
    }

    public Map<String, Object> getBeliefsSnapshot() {
        return beliefsSnapshot;
    }

    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{")
          .append("\"timestamp\":\"").append(timestamp).append("\",")
          .append("\"action\":\"").append(escape(action)).append("\",")
          .append("\"input\":\"").append(escape(input)).append("\",")
          .append("\"result\":\"").append(escape(result)).append("\",")
          .append("\"beliefs\":").append(beliefsToJson())
          .append("}");
        return sb.toString();
    }

    private String beliefsToJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : beliefsSnapshot.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            sb.append("\"").append(escape(e.getKey())).append("\":\"").append(escape(String.valueOf(e.getValue()))).append("\"");
        }
        sb.append("}");
        return sb.toString();
    }

    private String escape(String s) {
        if (s == null) return "";
        return s.replace("\"", "\\\"");
    }
}