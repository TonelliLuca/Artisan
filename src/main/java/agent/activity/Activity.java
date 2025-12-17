
package agent.activity;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class Activity {
    private final UUID uuid;
    private final String goal;
    private volatile Status status;
    private final List<ReasoningStep> history = new CopyOnWriteArrayList<>();
    private final Map<String, JsonNode> beliefs = new ConcurrentHashMap<>();
    private final List<JsonNode> incomingEvents = new CopyOnWriteArrayList<>();
    public enum Status {
        REASONING,
        ACTION,
        WAITING_FOR_EVENT,
        OBSERVATION,
        COMPLETED
    }

    public Activity(String goal) {
        this.uuid = UUID.randomUUID();
        this.goal = goal;
        this.status = Status.OBSERVATION;
    }

    public void pushEvent(JsonNode event) {
        incomingEvents.add(event);
    }

    public List<JsonNode> consumeEvents() {
        List<JsonNode> current = new ArrayList<>(incomingEvents);
        incomingEvents.clear();
        return current;
    }

    public boolean hasEvents() {
        return !incomingEvents.isEmpty();
    }

    public void setBelief(String key, JsonNode value) {
        if (value != null) {
            beliefs.put(key, value);
        }
    }

    public JsonNode getBelief(String key) {
        return beliefs.get(key);
    }
    public Map<String, Object> getBeliefsSnapshot() {
        return new HashMap<>(beliefs);
    }
    private String beliefsToJson() {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, JsonNode> entry : beliefs.entrySet()) {
            if (!first) sb.append(",");
            first = false;
            sb.append("\"").append(escape(entry.getKey())).append("\":").append(entry.getValue().toString());
        }
        sb.append("}");
        return sb.toString();
    }
    public String getUuid() {
        return uuid.toString();
    }

    public String getGoal() {
        return goal;
    }

    public Status getStatus() {
        return status;
    }

    public void setStatus(Status status) {
        this.status = status;
    }

    public boolean isCompleted() {
        return this.status == Status.COMPLETED;
    }

    public void addStep(ReasoningStep step) {
        if (step != null) {
            history.add(step);
        }
    }

    public List<ReasoningStep> getHistory() {
        return Collections.unmodifiableList(history);
    }

    public Optional<ReasoningStep> lastStep() {
        if (history.isEmpty()) return Optional.empty();
        return Optional.of(history.get(history.size() - 1));
    }

    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{")
          .append("\"uuid\":\"").append(uuid).append("\",")
          .append("\"goal\":\"").append(escape(goal)).append("\",")
          .append("\"status\":\"").append(status).append("\",")
            .append("\"variables\":").append(beliefsToJson()).append(",")
          .append("\"history\":[");
        boolean first = true;
        for (ReasoningStep step : history) {
            if (!first) sb.append(",");
            first = false;
            sb.append(step.toJson());
        }
        sb.append("]}");
        return sb.toString();
    }

    private String escape(String s) {
        if (s == null) return "";
        return s.replace("\"", "\\\"");
    }
}