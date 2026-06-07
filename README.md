# personalai

A local AI reasoning agent that runs entirely on-device using [exo](https://github.com/exo-explore/exo) for inference. No OpenAI API calls are made — the OpenAI SDK is used purely as an HTTP client pointing at `localhost:52415`.

## Stack

- **LangGraph** — orchestrates the agent as a stateful graph
- **Instructor** — wraps the LLM client to enforce structured JSON output
- **Pydantic** — defines the schema every model response must match
- **DuckDuckGo Search (`ddgs`)** — real web search, no API key required
- **exo** — local inference server (runs Gemma 4 31B on-device)

## How it works

The agent follows a ReAct-style loop:

```
[agent] → final_answer → END
[agent] → other tool  → [tools] → [agent] (loop)
```

1. The model receives the user's question and decides which tool to call
2. The tool runs and its result is fed back into the conversation
3. The model loops until it decides it has enough information to give a `final_answer`

## Code walkthrough (`agent.py`)

### Lines 2–8 — Imports

- `operator` — standard library, used for `operator.add` to merge lists in state
- `instructor` — wraps the LLM client to enforce structured JSON output
- `OpenAI` — used as an HTTP client to talk to the local exo server
- `BaseModel, Field` — Pydantic, defines the shape of the model's output
- `Annotated, TypedDict` — typing tools for defining the graph state
- `StateGraph, END` — LangGraph's graph builder and termination sentinel
- `DDGS` — DuckDuckGo search library

### Lines 13–16 — `AgentAction`

The schema every model response must match. Three fields:

- `reasoning` — why the model chose this action
- `tool_name` — which tool to call (`search`, `calculator`, `final_answer`)
- `query` — what to pass to the tool, or the final answer text

### Lines 22–28 — LLM client

Creates a client pointing at exo on `localhost:52415`. `instructor` wraps it so every `.create()` call automatically validates the response against `AgentAction`. `MD_JSON` mode tells instructor to look for JSON inside markdown code blocks in the response.

### Lines 33–37 — `State`

The shared data bag passed between all graph nodes:

- `input` — the original question
- `messages` — full conversation history; `operator.add` means new messages append instead of overwrite
- `action` — the model's last `AgentAction` decision
- `final_response` — the agent's final answer, written when done

### Lines 40–71 — `call_model` node

Called every time the agent needs to think. On the first run it seeds the conversation with a system prompt + user question. On subsequent runs the history already exists. Calls the model, gets back an `AgentAction`, records it in history. If the model chose `final_answer`, also writes `final_response` to state here (because the graph skips `execute_tool` for `final_answer`).

### Lines 73–91 — `execute_tool` node

Runs whichever tool the model picked:

- `search` — queries DuckDuckGo, joins top 3 results into a string
- anything else — returns "tool not found"

The result is fed back into the conversation as a user message so the model sees it on the next loop.

### Lines 95–98 — `should_continue`

The router. Returns `"end"` if the model chose `final_answer`, otherwise `"continue"` to loop back through tools.

### Lines 100–120 — Graph wiring

Builds and compiles the graph with conditional edges based on the router above.

### Lines 123–125 — Run

Invokes the compiled graph with a question and prints the final answer.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install instructor openai langgraph pydantic ddgs
```

Make sure exo is running locally with a Gemma model loaded before running:

```bash
python3 agent.py
```

below is the whole agent file

```python

import operator
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from ddgs import DDGS

# --- 1. Structured Output Schema ---
# Defines the shape of every decision the model makes.
# instructor enforces the model always returns valid JSON matching this schema.
class AgentAction(BaseModel):
    reasoning: str = Field(description="The thought process behind the action")
    tool_name: str = Field(description="The tool to use: 'search', 'calculator', or 'final_answer'")
    query: str = Field(description="The input for the tool or the final response")

# --- 2. Local LLM Setup ---
# The OpenAI SDK is used purely as an HTTP client — no OpenAI API calls are made.
# It points to a local exo inference server running on port 52415.
# instructor wraps it to enforce structured JSON output via response_model.
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:52415/v1",
        api_key="exo"  # placeholder — local servers don't validate API keys
    ),
    mode=instructor.Mode.MD_JSON  # tells instructor to extract structured output from raw JSON
)

# --- 3. LangGraph State ---
# Shared state passed between all nodes in the graph.
# `operator.add` on `messages` means new messages are appended rather than overwriting.
class State(TypedDict):
    input: str                              # the original user question
    messages: Annotated[list, operator.add] # full conversation history
    action: AgentAction                     # the model's last decision
    final_response: str                     # set when the agent is done

# --- 4. Nodes ---
def call_model(state: State):
    # On the first run, seed the conversation with a system prompt and the user question.
    # On subsequent runs, messages already contain the full history.
    current_messages = state.get("messages", [])
    if not current_messages:
        current_messages = [
            {"role": "system", "content": "You are a helpful reasoning agent. Use tools to find information. Available tools: 'search', 'calculator', 'final_answer'."},
            {"role": "user", "content": state["input"]}
        ]

    # Ask the model what to do next — returns a structured AgentAction
    response = client.create(
        model="mlx-community/gemma-4-31b-it-8bit",
        messages=current_messages, #type: ignore
        response_model=AgentAction, #type: ignore
        max_retries=2  # <-- ADD THIS: Crash immediately if the JSON is bad
    )

    # Record the model's decision in history so future turns have full context
    new_message = {
        "role": "assistant",
        "content": f"Action: {response.tool_name} | Query: {response.query} | Reasoning: {response.reasoning}"
    }

    result = {"action": response, "messages": [new_message] if state.get("messages") else current_messages + [new_message]}

    # If the model chose final_answer, write the response directly to state here.
    # The graph routes final_answer straight to END (skipping execute_tool),
    # so final_response must be set in this node, not execute_tool.
    if response.tool_name == "final_answer":
        result["final_response"] = response.query
    return result

def execute_tool(state: State):
    action = state["action"]

    # Actual tool implementations — replace with real logic as needed
    if action.tool_name == "search":
        with DDGS() as ddgs:
            results = list(ddgs.text(action.query, max_results=3))
            if not results:
                result = "No results found for the query."
            else:
                result = "\n".join(f"{r['title']}: {r['body']}" for r in results)
        result = "\n".join(f"{r['title']}: {r['body']}" for r in results)
    else:
        result = f"Tool '{action.tool_name}' not found."

    # Feed the tool result back into the conversation so the model can act on it
    tool_feedback = {"role": "user", "content": f"Tool observation: {result}"}

    return {"messages": [tool_feedback]}

# --- 5. Graph Logic ---
# Decides whether to loop back through tools or stop.
def should_continue(state: State):
    if state["action"].tool_name == "final_answer":
        return "end"
    return "continue"

workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tool)

workflow.set_entry_point("agent")

# agent → final_answer → END
# agent → any other tool → tools → agent (loop)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- 6. Run ---
final_state = app.invoke({"input": "What is the date today in Tokyo?", "messages": []}) #type: ignore
print("\n--- Final Output ---")
print(final_state["final_response"])
```
