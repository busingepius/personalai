
import operator
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

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
    mode=instructor.Mode.JSON  # tells instructor to extract structured output from raw JSON
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

    # Mock tool implementations — replace with real logic as needed
    if action.tool_name == "search":
        result = f"Search result for '{action.query}': Sunny, 75°F"
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
final_state = app.invoke({"input": "What is the weather in Tokyo?", "messages": []}) #type: ignore
print("\n--- Final Output ---")
print(final_state["final_response"])
