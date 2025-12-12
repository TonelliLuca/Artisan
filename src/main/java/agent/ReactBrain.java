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
        RECENT HISTORY:
        {{history}}
        Context: {{context}}

        Think step-by-step and output a structured reasoning result.
   
        Rules:
            - IMPORTANT NEVER call tools in this phase and never do function call.
            - ALWAYS return a brief reasoning summary.
            - DO NOT return anything other than the reasoning summary.
            - IMPORTANT - this summary will be used in the next phases.
    """)
    String reason(@V("goal")String goal, @V("history")String history, @V("context")String context);

    @UserMessage("""
        You are the ACTION phase.
        Goal: {{goal}}
        RECENT HISTORY:
        {{history}}
        Context: {{context}}

        Perform the necessary action using your available tools. If no tool is useful, you may decide not to act and must not call any tool.        
        AFTER acting (or deciding not to act), return a JSON summary:
        {
          "tool_name": "The name of the tool you used (or null)",
          "summary": "Brief result of the action"
        }
        
        IMPORTANT:
        - If you called a tool, 'tool_name' MUST be populated.
        - This signals the system to wait for asynchronous events (SSE).
    """)
    String act(@V("goal")String goal, @V("history")String history, @V("context")String context);

    @UserMessage("""
    You are the OBSERVATION phase.
    
    Goal: {{goal}}
    Context Events: {{events}}
    RECENT HISTORY:
    {{history}}
    Context: {{context}}

    YOUR PRIORITY TASK:
    Analyze the 'Context Events' list provided above.
    
    1. SEARCH specifically for completion events (like 'timer.finished', 'alert', etc.).
    2. IF you see 'timer.finished' (or similar) inside Context Events -> THE GOAL IS COMPLETED.
    3. IGNORE the internal state of variables if the Event says it is finished. Events are the source of truth.

    Return ONLY JSON:
    {
      "completed": true|false,
      "summary": "FOUND event X, so completed / NO event found, waiting",
      "beliefs": { ... }
    }
""")
    String observe(@V("goal") String goal, @V("history") String history, @V("context") String context, @V("events") String events);
}