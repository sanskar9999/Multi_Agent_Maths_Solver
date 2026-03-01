# AIMO Multi-Agent Solver 🧠🚀

An autonomous mathematical reasoning engine that solves competition-level math problems using a specialized multi-agent pipeline and real-time code execution.

<img width="960" height="540" alt="MULTI_AGENT_MATH_SOLVER" src="https://github.com/user-attachments/assets/84afd8b1-7f78-4b48-aa0f-a39b0294eddd" />


## 🌟 Overview

The **AIMO Multi-Agent Solver** replaces traditional single-prompt reasoning with a robust, code-enforced pipeline. By breaking down problems into atomic steps and executing Python code for every calculation, it eliminates hallucinations and ensures mathematical rigor.

### 🤖 The Multi-Agent Pipeline

Behind the scenes, four specialized micro-agents collaborate under a central Python Orchestrator:

1.  **🧩 Decomposer**: Analyzes the current state and identifies the **one next miniscule computation** required. It prevents the model from attempting to solve everything at once.
2.  **💻 Coder**: Writes a focused Python snippet (5-30 lines) to achieve exactly what the Decomposer requested. No mental math allowed.
3.  **📊 Sandbox Execution**: Runs the code in a secure Jupyter kernel and captures the raw output.
4.  **🔍 Interpreter**: Analyzes the code output, updates the "scratchpad," and decides if the problem is solved or needs another step.
5.  **🛡️ Verifier**: Once a final answer is proposed, this agent independently re-checks the solution from multiple different mathematical angles using fresh code.

## 🚀 Key Features

*   **Atomic Step Reasoning**: Prevents lengthy, hallucinatory comments by forcing small code-run cycles.
*   **Real-Time UI**: A premium, dark-mode dashboard built with React and Tailwind CSS v4, featuring a live timeline of agent actions.
*   **Secure Execution**: All code runs in an isolated Jupyter environment.
*   **Streaming Logic**: Watch the AI "think" and "write code" in real-time via WebSockets.
*   **Zero Mental Math**: Every calculation is performed by Python, ensuring 100% accuracy in arithmetic and symbolic operations.

## 🛠️ Technology Stack

*   **Logic**: Python, OpenAI API (compatible with Ollama/Local LLMs).
*   **Math**: SymPy, NumPy, Itertools.
*   **Backend**: FastAPI, Uvicorn, WebSockets.
*   **Frontend**: React, Vite, Tailwind CSS v4, Framer Motion, Lucide Icons.

## 📥 Installation & Setup

### 1. Prerequisites
*   Python 3.10+
*   Node.js & npm
*   (Optional) Ollama running `svjack/Qwen3-4B-Instruct-2507-heretic` or similar.

### 2. Backend Setup
```bash
cd aimo_ui/backend
pip install -r requirements.txt  # Ensure fastapi, uvicorn, openai, sympy, pandas are installed
python main.py
```

### 3. Frontend Setup
```bash
cd aimo_ui/frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to start solving.

## 📜 Problem Sets
The solver reads from `reference.csv`, which contains various competition-style math problems. You can also input custom problems directly in the UI.

---
Developed by Sanskar Verma.

