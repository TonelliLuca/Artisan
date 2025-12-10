
package agent.activity;

import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.Collections;

public class Activity {
    private final UUID uuid;
    private final String goal;
    private volatile Status status;
    private final List<ReasoningStep> history = new CopyOnWriteArrayList<>();

    public enum Status {
        REASONING,
        ACTION,
        OBSERVATION,
        COMPLETED
    }

    public Activity(String goal) {
        this.uuid = UUID.randomUUID();
        this.goal = goal;
        this.status = Status.REASONING;
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