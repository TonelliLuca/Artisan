package agent;

import dev.langchain4j.agentic.Agent;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface ReactBrain {

    @Agent("You are the BRAIN of a reactive agent that follows the REASON-ACT-OBSERVE loop. solve problems step-by-step by reasoning about them, deciding on actions, and observing results." +
            "you can use tools to perform actions as needed but only in the ACTION phase.")
    @UserMessage("""
        You are the REASONING phase of a reactive agent.
        Goal: {{goal}}
        Last step: {{lastStep}}
        Context: {{context}}

        Think step-by-step and output a structured reasoning result.
   
        Rules:
            - NEVER call tools in this phase.
    """)
    String reason(@V("goal")String goal, @V("lastStep")String lastStep, @V("context")String context);

    @UserMessage("""
        You are the ACTION phase of a reactive agent.
        Goal: {{goal}}
        Last step: {{lastStep}}
        Context: {{context}}

        Decide and describe the exact action to take (tool call or passing).
    """)
    String act(@V("goal")String goal, @V("lastStep")String lastStep, @V("context")String context);

    @UserMessage("""
    You are the OBSERVATION phase of a reactive agent.

    Goal: {{goal}}
    Last step: {{lastStep}}
    Context: {{context}}

    IMPORTANT:
    You must return ONLY a valid JSON object as plain text.
    No explanation. No prose. No commentary.

    The JSON MUST have exactly these fields:
    {
      "completed": true|false,
      "summary": "short string",
      "beliefs": { ... }   // or an empty object {}
    }

    Rules:
    - completed=true if the goal is achieved OR clearly impossible.
    - completed=false if the agent should keep reasoning.
    - beliefs is optional but must ALWAYS be an object ({} if empty).
    - NEVER call tools in this phase.
    - NEVER return anything that isn't valid JSON.
    - NEVER return XML, markdown, or code blocks.
    - ONLY the JSON object, nothing else.
""")
    String observe(@V("goal") String goal, @V("lastStep") String lastStep, @V("context") String context);
}