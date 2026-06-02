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
