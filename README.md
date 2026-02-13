# Iterative Self-Correcting Agent (LangGraph)

Project Overview

This project implements an **iterative autonomous agent** that can:
generate code
review its own output
detect issues
revise the solution
repeat until quality criteria are met

Unlike a simple LLM call, this system performs **multi-step reasoning with self-correction** using a cyclic graph workflow.


 How It Works

The agent is built with a structured state graph consisting of three core nodes:

 1️-Writer

Generates complete Python code from a natural language task.

2️ -Critic

Analyzes the code and evaluates:

correctness
edge cases
readability
security

Outputs structured feedback and a verdict:

```
PASS
or
NEEDS_REVISION
```

### 3️- Editor

Fixes all issues identified by the Critic and produces an improved version of the code.


 Iterative Loop Logic

Workflow:

```
User Task
   ↓
Writer → Critic → Editor → Critic → ... → Final Output
```

The loop continues until:

* the code passes review
* OR max iterations reached



 Agent Properties

 autonomous reasoning loop
 self-critique
 iterative refinement
 structured decision routing
 deterministic outputs
 state-based execution

How to Run

Set API key

```
export OPENAI_API_KEY="your-key"
```

Run agent

```
python AgentAI.py
```

Then enter any task:

```
Task: bubble sort
```




Goal of the Project

The goal is to demonstrate an **iterative reasoning agent architecture** instead of a single LLM call.

The system simulates a real engineering workflow:

> write → review → fix → repeat → deliver



## 🏆 Educational Value

This project demonstrates concepts used in modern AI systems:

  agentic workflows
  structured prompting
  reasoning loops
  self-evaluation systems
  autonomous decision making


License

Educational use.
