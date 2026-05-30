
import operator
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

# --- 1. Structured Output Schema ---
class AgentAction(BaseModel):
    reasoning: str = Field(description="The thought process behind the action")
    tool_name: str = Field(description="The tool to use: 'search', 'calculator', or 'final_answer'")
    query: str = Field(description="The input for the tool or the final response")

# --- 2. Local LLM Setup ---
client = instructor.from_openai(
    OpenAI(
        # MLX server defaults to port 8080
        base_url="http://localhost:8080/v1", 
        api_key="mlx" 
    ),
    mode=instructor.Mode.JSON 
)

# --- 3. LangGraph State ---
# FIX: Use `operator.add` to append new messages to the history list automatically,
# rather than overwriting the state every time.
class State(TypedDict):
    input: str
    messages: Annotated[list, operator.add]
    action: AgentAction
    final_response: str

# --- 4. Nodes ---
def call_model(state: State):
    # Initialize messages on the first run
    current_messages = state.get("messages", [])
    if not current_messages:
        current_messages = [
            {"role": "system", "content": "You are a helpful reasoning agent. Use tools to find information. Available tools: 'search', 'calculator', 'final_answer'."},
            {"role": "user", "content": state["input"]}
        ]

    response = client.chat.completions.create(
        # Pass the exact MLX repo name
        model="mlx-community/gemma-4-31b-it-8bit", 
        messages=current_messages,
        response_model=AgentAction,
    )

    # FIX: Add the agent's decision to the conversation history so it remembers what it did
    new_message = {
        "role": "assistant",
        "content": f"Action: {response.tool_name} | Query: {response.query} | Reasoning: {response.reasoning}"
    }

    # Because of `operator.add` in the State, returning a list here APPENDS it.
    return {"action": response, "messages": [new_message] if state.get("messages") else current_messages + [new_message]}

def execute_tool(state: State):
    action = state["action"]

    # Mock Logic for tool execution
    if action.tool_name == "search":
        result = f"Search result for '{action.query}': Sunny, 75°F"
    elif action.tool_name == "final_answer":
        return {"final_response": action.query}
    else:
        result = f"Tool '{action.tool_name}' not found."

    # FIX: Add the tool's result to the conversation history as a new message
    tool_feedback = {"role": "user", "content": f"Tool observation: {result}"}

    return {"messages": [tool_feedback]}

# --- 5. Graph Logic ---
def should_continue(state: State):
    if state["action"].tool_name == "final_answer":
        return "end"
    return "continue"

workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tool)

workflow.set_entry_point("agent")

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
final_state = app.invoke({"input": "What is the weather in Tokyo?", "messages": []})
print("\n--- Final Output ---")
print(final_state["final_response"])
