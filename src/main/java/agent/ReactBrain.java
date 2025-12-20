package agent;

import dev.langchain4j.agentic.Agent;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

public interface ReactBrain {

    @Agent("""
        You are the BRAIN of an asynchronous, event-driven agent.
        
        SYSTEM ARCHITECTURE (CRITICAL):
        1. ASYNCHRONOUS TOOLS: Your tools are non-blocking. Calling a tool only INITIATES an action (e.g., "Start Timer").
        2. SSE EVENTS: The actual result or completion arrives later as an asynchronous Event via SSE.
        3. BELIEF UPDATES: These events automatically update your 'CONTEXT' (Variables/Beliefs).
        4. HISTORY IS KEY: Since actions are split from results, you must strictly check 'HISTORY' to know if you have already started an action.
        
        Follow the REASON-ACT-OBSERVE loop to solve problems step-by-step.
    """)
    @UserMessage("""
        You are the REASONING phase.
        
        MAIN GOAL: {{goal}}
        === RELEVANT PAST MEMORIES (Use these as a guide) ===
                        {{memories}}
        ============================================================
        === PROGRESS TRACKER (The Master Plan) ===
        {{progress}}
        ==========================================
       
        
        CURRENT CONTEXT (Variables): {{context}}

        YOUR TASK:
        1. Check PAST MEMORIES first! Does a successful procedure exist?
        2. IF YES: You MUST follow the procedure steps exactly, even if they seem redundant (e.g. subscribing first).
        3. Look at the PROGRESS TRACKER. Identify what is marked [x] (Done) and what is [ ] (Pending).
        4. Decide the very next step based on the first Pending item and the context, the context beliefs are very IMPORTANT.
        5. Formulate a plan for the ACTION phase.

        IMPORTANT RULES:
        - NEVER call tools in this phase.
        - NEVER output function calls here.
        - ALWAYS return a brief reasoning summary text.
        - DO NOT return anything other than the reasoning summary.
        
        RECENT HISTORY (Last 5 steps):
        {{history}}
    """)
    String reason(@V("goal")String goal, @V("history")String history, @V("context")String context, @V("progress") String progress, @V("memories") String memories);

    @UserMessage("""
        You are the ACTION phase, if i ask to do a tool call you need to perform it via remote procedure call.
        
        YOUR TASK:
        Execute the next pending action using the available tools or do nothing if no tool is applicable.
        Use the provided tools only if they correspond to the next pending step in the PROGRESS TRACKER.
        If you decide to use a tool call the tool **natively** with the correct parameters.
        If no tool is applicable, do NOT call any tool and explain why in the summary.
        
        GOAL: {{goal}}
        PROGRESS: {{progress}}
        CONTEXT: {{context}}
        


        STRICT TOOL USE RULES:
            1. You have access to specific tools (e.g., 'timerTool').
            2. ONLY use a tool if it directly solves the current step of the Goal.
            3. DO NOT use tools "just in case" or for unrelated tasks.
            4. If the available tools do not match the Goal, YOU MUST NOT CALL ANY TOOL.
               Instead, return "tool_name": null and a summary explaining why.
        
        !!! ANTI-LOOP SAFEGUARDS !!!
        1. Tools ONLY accept the parameters defined in their schema.
        2. Use the tool ONLY ONCE per turn.
        
        Expected Format only JSON (no markdown, no code blocks, no extra text):
        {
          "tool_name": "The name of the tool you used (or null)",
          "summary": "Brief result of the action"
        }
        
        IMPORTANT:
        - If you called a tool, 'tool_name' MUST be populated.
        - This signals the system to wait for asynchronous events (SSE).
        
        RECENT HISTORY (Last 5 steps):
        {{history}}

    """)
    String act(@V("goal")String goal, @V("history")String history, @V("context")String context, @V("progress") String progress);

    @UserMessage("""
    You are the OBSERVATION phase.
    
    GOAL: {{goal}}
    
    CURRENT PROGRESS TRACKER:
    {{progress}}
    
    CONTEXT EVENTS: {{events}}
    CONTEXT: {{context}}

    YOUR PRIORITY TASK:
    Analyze the 'Context Events' and 'HISTORY' and 'CONTEXT' to update the plan or create a new one.
    
    1. INITIAL PLANNING (If Progress is empty):
       - Assess if the GOAL is achievable with the tools you likely have or general logic.
       - IF ACHIEVABLE: Create the initial Master Plan in 'new_progress' and set each point to [ ], Break down complex goals into distinct, atomic steps.
       - IF IMPOSSIBLE: Do NOT create a plan. Set "completed": true and explain why in "summary".

    
    2. PROGRESS UPDATE (STRICT VERIFICATION):
       - You may ONLY mark a pending item [ ] as [x] IF:
         a) An event in 'CONTEXT EVENTS' explicitly confirms it (e.g., 'timer.finished').
         b) The 'HISTORY' shows you just successfully performed the Action for that step.
       - DO NOT mark a step as [x] just because previous steps are done.
       - If you have NOT performed the action for a pending step yet, keep it as [ ].

       
    3. COMPLETION CHECK:
       If the entire Goal is achieved based on the tracker (all items are [x]), YOU MUST return "completed": true IMMEDIATELY.
    
    4. FAILURE/IMPOSSIBILITY CHECK:
       Look at the 'HISTORY'. Did the 'ACTION' phase fail to find a tool or report an error?
       If so, mark "completed": true (giving up is a valid completion).

    CRITICAL OUTPUT RULES:
    1. Return STRICTLY RAW JSON.
    2. DO NOT use Markdown code blocks.
    3. Start the response immediately with '{'.
    4. DO NOT pass this JSON summary to the tool itself!

    Return JSON format:
    {
      "completed": true|false,
      "summary": "Explanation of what happened",
      
      "new_progress": "1 [ ] Step 1\\n 2 [ ] Step 2...", 
      
      "update_variables": { ... }
    }
    IMPORTANT RULES:
        - NEVER call tools in this phase.
        - NEVER output function calls here.
        
    RECENT HISTORY (Last 5 steps):
    {{history}}

""")
    String observe(@V("goal") String goal, @V("history") String history, @V("context") String context, @V("events") String events, @V("progress") String progress);

    @UserMessage("""
        You are the REFLECTION phase.
        
        TASK GOAL: {{goal}}
        FINAL STATUS: {{status}}
        FULL HISTORY: 
        {{history}}
        
        Your job is to compress this experience into a reusable memory for future reference.
         Analyze the history to understand what went well and what didn't.
        
        1. SUMMARY: Briefly describe what was done to achieve (or fail) the goal.
        2. LESSONS: Did anything fail? How was it fixed? If it was impossible, why?
        3. PROCEDURE: Extract the high-level steps that worked (The "Happy Path"). 
           - Generalize specific values (e.g., instead of "set timer for 5s", use "set timer for required duration").
           - Note any important dependencies (e.g., "Wait for X before doing Y").
        
        CRITICAL: Return ONLY JSON. No markdown.
        
        Return JSON format:
        {
          "summary": "Brief summary of the activity",
          "outcome": "SUCCESS" or "FAILURE",
          "lessons_learned": ["Lesson 1", "Lesson 2"],
          "successful_procedure": ["Step 1", "Step 2", "Step 3"],
          "keywords": ["keyword1", "keyword2"]
        }
    """)
    String reflect(@V("goal") String goal, @V("status") String status, @V("history") String history);
}