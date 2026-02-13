"""
Iterative Self-Correcting Code Reviewer Agent
Workflow: WRITE -> CRITIQUE -> EDIT -> (repeat if needed) -> FINAL OUTPUT

Usage:
    python self_correcting_agent.py
    Then enter any natural language task when prompted.
"""

import os
import sys
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


# ====================================================
# AGENT STATE
# ====================================================

class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the graph.

    Fields:
        user_task       : The raw natural language task entered by the user at runtime.
        current_code    : The latest version of generated/revised code.
        critique_feedback: Full structured feedback produced by the Critic node.
        verdict         : Current review verdict — "INITIAL", "PASS", or "NEEDS_REVISION".
        iteration_count : How many full Critic passes have completed so far.
    """
    user_task: str
    current_code: str
    critique_feedback: str
    verdict: Literal["PASS", "NEEDS_REVISION", "INITIAL"]
    iteration_count: int


# ====================================================
# LLM SETUP
# ====================================================

# Requires OPENAI_API_KEY to be set in the environment.
# Temperature is low to keep outputs deterministic and structured.
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# ====================================================
# HELPER: STRIP MARKDOWN CODE FENCES
# ====================================================

def strip_code_fences(text: str) -> str:
    """
    Removes markdown triple-backtick fences from LLM output.
    Handles ```python ... ```, ``` ... ```, and plain text equally.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove the opening fence line (e.g. ```python or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove the closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


# ====================================================
# NODE 1: WRITER
# ====================================================

def writer_node(state: AgentState) -> AgentState:
    """
    Reads the user's natural language task from state,
    designs a solution, and generates complete, runnable Python code.

    This node only executes once — on the very first pass.
    The code it produces must be full, not partial, not pseudo-code.
    """
    print()
    print("=" * 60)
    print("WRITER NODE  —  Generating initial solution...")
    print("=" * 60)

    user_task = state["user_task"]

    system_prompt = (
        "You are a senior Python software engineer.\n"
        "Your job is to:\n"
        "  1. Read the user's task carefully.\n"
        "  2. Design an appropriate solution in your head.\n"
        "  3. Write complete, fully working Python code that solves the task.\n\n"
        "Rules you MUST follow:\n"
        "  - Output ONLY raw Python code. No explanations. No markdown. No prose.\n"
        "  - The code must be complete and immediately runnable as-is.\n"
        "  - Include all necessary imports.\n"
        "  - Use clear variable names and inline comments.\n"
        "  - Handle edge cases (empty inputs, None, type errors, boundary values).\n"
        "  - Never use forbidden constructs if the task explicitly forbids them.\n"
        "  - Include a __main__ block with representative test cases that print results.\n"
    )

    user_prompt = (
        f"Task:\n{user_task}\n\n"
        "Generate the complete Python solution now. Output only raw code."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    code = strip_code_fences(response.content)

    print()
    print("Generated code:")
    print("-" * 60)
    print(code)
    print("-" * 60)

    return {
        **state,
        "current_code": code,
        "verdict": "INITIAL",
        "iteration_count": 0,
    }


# ====================================================
# NODE 2: CRITIC
# ====================================================

def critic_node(state: AgentState) -> AgentState:
    """
    Rigorously reviews the current code across four dimensions:
      - Correctness   : Does the logic solve the stated task correctly?
      - Edge cases    : Are boundary/null/empty/invalid inputs handled?
      - Readability   : Is the code clear, well-named, and well-commented?
      - Security      : Are there injection risks, unsafe evals, path traversal, etc.?

    Outputs a strictly formatted verdict block so routing logic can parse it.
    """
    iteration = state["iteration_count"] + 1
    print()
    print("=" * 60)
    print(f"CRITIC NODE  —  Review pass #{iteration}")
    print("=" * 60)

    user_task = state["user_task"]
    current_code = state["current_code"]

    system_prompt = (
        "You are a strict, senior code reviewer with expertise in Python correctness, "
        "security, and software craftsmanship.\n\n"
        "You will review code against the original task and score it across four dimensions. "
        "You must be thorough and uncompromising. Do not give PASS unless ALL four dimensions "
        "are fully satisfied.\n\n"
        "Your response MUST use this exact format with no deviations:\n\n"
        "VERDICT: <PASS or NEEDS_REVISION>\n\n"
        "CORRECTNESS:\n"
        "<Assessment. Note any logical errors, wrong algorithms, or incorrect output.>\n\n"
        "EDGE_CASES:\n"
        "<Assessment. Note any unhandled inputs: empty list, None, negatives, duplicates, "
        "large inputs, type mismatches, etc.>\n\n"
        "READABILITY:\n"
        "<Assessment. Note unclear names, missing comments, overly complex logic, "
        "non-Pythonic style, missing docstrings.>\n\n"
        "SECURITY:\n"
        "<Assessment. Note any use of eval/exec on user input, shell injection, "
        "unvalidated file paths, unsafe deserialization, hardcoded secrets, etc. "
        "If none apply, state: No security issues detected.>\n\n"
        "ISSUES:\n"
        "<Bullet list of every specific issue that must be fixed. "
        "If verdict is PASS, write: None.>\n\n"
        "Do not output anything outside this format."
    )

    user_prompt = (
        f"Original task:\n{user_task}\n\n"
        f"Code to review:\n{current_code}\n\n"
        "Provide your structured review now."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    critique = response.content.strip()

    # Parse verdict — look for the canonical VERDICT: line
    verdict = "NEEDS_REVISION"  # safe default
    for line in critique.splitlines():
        stripped = line.strip()
        if stripped.startswith("VERDICT:"):
            value = stripped.split(":", 1)[1].strip().upper()
            if value == "PASS":
                verdict = "PASS"
            else:
                verdict = "NEEDS_REVISION"
            break

    print()
    print("Critique:")
    print("-" * 60)
    print(critique)
    print("-" * 60)
    print(f"\nVerdict: {verdict}")

    return {
        **state,
        "critique_feedback": critique,
        "verdict": verdict,
        "iteration_count": state["iteration_count"] + 1,
    }


# ====================================================
# NODE 3: EDITOR
# ====================================================

def editor_node(state: AgentState) -> AgentState:
    """
    Receives the Critic's structured feedback and produces a revised version
    of the code that addresses every single issue mentioned.

    Rules enforced via prompt:
      - Fix ALL issues listed under ISSUES in the critique.
      - Do not introduce new bugs while fixing existing ones.
      - Output ONLY raw Python code — no explanations, no prose, no markdown.
    """
    print()
    print("=" * 60)
    print("EDITOR NODE  —  Applying revisions...")
    print("=" * 60)

    user_task = state["user_task"]
    current_code = state["current_code"]
    critique = state["critique_feedback"]

    system_prompt = (
        "You are a meticulous Python engineer responsible for fixing code based on a "
        "formal code review.\n\n"
        "Rules you MUST follow:\n"
        "  - Read every section of the critique (CORRECTNESS, EDGE_CASES, READABILITY, "
        "SECURITY, ISSUES) and fix every single problem described.\n"
        "  - Do not skip any issue, no matter how minor.\n"
        "  - Do not introduce new bugs while fixing existing ones.\n"
        "  - Preserve the overall structure and intent of the original code.\n"
        "  - Output ONLY the revised raw Python code.\n"
        "  - No explanations. No markdown. No comments about what you changed. "
        "Only the code.\n"
    )

    user_prompt = (
        f"Original task:\n{user_task}\n\n"
        f"Current code (to be revised):\n{current_code}\n\n"
        f"Code review that must be addressed:\n{critique}\n\n"
        "Output the fully revised Python code now. Raw code only."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    revised_code = strip_code_fences(response.content)

    print()
    print("Revised code:")
    print("-" * 60)
    print(revised_code)
    print("-" * 60)

    return {
        **state,
        "current_code": revised_code,
    }


# ====================================================
# ROUTING LOGIC
# ====================================================

def route_after_critic(state: AgentState) -> Literal["editor", "end"]:
    """
    Determines what happens after the Critic node runs.

    Rules:
      - If verdict is PASS             → end the graph (success).
      - If iteration_count >= 3        → end the graph (max iterations reached).
      - If verdict is NEEDS_REVISION   → send to Editor for another cycle.
    """
    if state["verdict"] == "PASS":
        print("\nRouting decision: PASS — ending workflow.")
        return "end"

    if state["iteration_count"] >= 3:
        print("\nRouting decision: max iterations (3) reached — ending workflow.")
        return "end"

    print("\nRouting decision: NEEDS_REVISION — sending to Editor.")
    return "editor"


# ====================================================
# GRAPH CONSTRUCTION
# ====================================================

def build_graph() -> StateGraph:
    """
    Assembles the LangGraph state machine.

    Graph topology:
        [START] --> writer --> critic --+-- (PASS or max iters) --> [END]
                                        |
                                    (NEEDS_REVISION)
                                        |
                                     editor --> critic  (cycle)
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("editor", editor_node)

    # Entry point
    graph.set_entry_point("writer")

    # writer → critic (always, first pass)
    graph.add_edge("writer", "critic")

    # critic → editor OR end (conditional)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "editor": "editor",
            "end": END,
        },
    )

    # editor → critic (always, back for re-review)
    graph.add_edge("editor", "critic")

    return graph.compile()


# ====================================================
# MAIN EXECUTION
# ====================================================

def main() -> None:
    """
    Entry point. Prompts the user for a task at runtime, runs the agent,
    and prints the final code, verdict, and iteration count.
    """
    print()
    print("=" * 60)
    print(" ITERATIVE SELF-CORRECTING CODE REVIEWER AGENT")
    print("=" * 60)
    print()
    print("Enter the task you want the agent to solve.")
    print("Be as specific or as general as you like.")
    print()

    # Runtime task input — no hardcoded task
    user_task = input("Task: ").strip()

    if not user_task:
        print("No task entered. Exiting.")
        sys.exit(1)

    print()
    print(f"Task received: {user_task}")

    # Build the compiled graph
    app = build_graph()

    # Seed the initial state
    initial_state: AgentState = {
        "user_task": user_task,
        "current_code": "",
        "critique_feedback": "",
        "verdict": "INITIAL",
        "iteration_count": 0,
    }

    print("\nStarting agent workflow...\n")

    # Run the graph to completion
    final_state: AgentState = app.invoke(initial_state)

    # ------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------
    print()
    print("=" * 60)
    print(" FINAL OUTPUT")
    print("=" * 60)
    print()
    print(f"Iterations completed : {final_state['iteration_count']}")
    print(f"Final verdict        : {final_state['verdict']}")
    print()
    print("Final code:")
    print("-" * 60)
    print(final_state["current_code"])
    print("-" * 60)
    print()
    print("Final critique:")
    print("-" * 60)
    print(final_state["critique_feedback"])
    print("-" * 60)
    print()
    print("Agent workflow finished.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print()
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Export it before running:")
        print("    export OPENAI_API_KEY='sk-...'")
        print()
        sys.exit(1)

    main()