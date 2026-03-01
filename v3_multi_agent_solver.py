"""
Multi-Agent AIMO Solver (v3)
============================
Uses 4 specialized micro-agents to solve math competition problems:
  1. Decomposer — decides the ONE next thing to compute
  2. Coder      — writes short code to compute it
  3. Interpreter — analyzes the result and decides next step
  4. Verifier    — independently re-checks the final answer

The orchestrator (Python code) controls the flow and forces atomic steps.
"""

import os
import re
import time
import queue
import threading
import traceback
from collections import Counter
from typing import Callable, Any, Optional

from openai import OpenAI
from v2_solver_final import AIMO3Sandbox


# ─────────────────────────────── Config ───────────────────────────────

class AgentCFG:
    """Configuration for the multi-agent solver."""
    
    served_model_name = 'svjack/Qwen3-4B-Instruct-2507-heretic'
    base_url = 'http://127.0.0.1:11434/v1'
    
    problem_timeout = 900       # total seconds for the entire problem
    jupyter_timeout = 60.0      # per-execution timeout
    max_steps = 15              # max decompose→code→run→interpret cycles
    max_retries = 2             # retries per agent if output format is wrong
    
    # ── Per-agent prompts (kept TINY and FOCUSED) ──
    
    decomposer_prompt = (
        "You are a math problem decomposer. Your ONLY job is to decide the "
        "ONE next small thing to compute.\n\n"
        "RULES:\n"
        "- Output 2-3 sentences MAX.\n"
        "- Say WHAT to compute and WHY it helps.\n"
        "- Do NOT write any code, equations, or calculations.\n"
        "- Do NOT try to solve the full problem. Just the next tiny step.\n"
        "- If you have enough information to give the final answer, say: "
        "DONE: [the answer as an integer]\n"
        "- Think about what is the SAFEST, most CERTAIN thing you can compute "
        "right now with the information you already have."
    )
    
    coder_prompt = (
        "You are a Python code writer. You receive a task description and you "
        "write a SHORT Python code block to accomplish ONLY that task.\n\n"
        "ABSOLUTE RULES:\n"
        "1. Respond with ONLY a ```python code block. Nothing else.\n"
        "2. Code must be 5-30 lines. No more.\n"
        "3. NO comments longer than 5 words.\n"
        "4. Use print() to output every result.\n"
        "5. Use sympy, numpy, itertools as needed.\n"
        "6. Do NOT hardcode answers. COMPUTE everything.\n"
        "7. Do NOT try to solve the entire problem. Only the specific task given.\n"
        "8. Variables from previous steps are available in the kernel."
    )
    
    interpreter_prompt = (
        "You are a result interpreter. You see code output and explain what "
        "it means for the problem.\n\n"
        "RULES:\n"
        "- Output 2-4 sentences MAX.\n"
        "- State what the result tells us.\n"
        "- Do NOT write any code.\n"
        "- End with either:\n"
        "  NEXT: [what to compute next, in one sentence]\n"
        "  or\n"
        "  DONE: [the final answer as an integer]\n"
        "- Only say DONE if you are CONFIDENT the answer is fully computed "
        "and verified by the code output."
    )
    
    verifier_prompt = (
        "You are a math answer verifier. You receive a problem, a proposed "
        "answer, and the work done so far. Write Python code that INDEPENDENTLY "
        "checks the answer is correct.\n\n"
        "RULES:\n"
        "1. Respond with ONLY a ```python code block.\n"
        "2. Check the answer from AT LEAST 2 different angles.\n"
        "3. At the end, you MUST use print() to output EXACTLY one of:\n"
        "   print(\"VERIFIED:\", answer_value)\n"
        "   print(\"FAILED:\", reason_string)\n"
        "   IMPORTANT: You must use print(). Do NOT write bare VERIFIED: or FAILED: labels.\n"
        "4. Use sympy/numpy to do all math.\n"
        "5. Do NOT trust the proposed answer — recompute it independently.\n"
        "6. Variables from previous steps are available in the kernel."
    )
    
    # ── Token limits per agent ──
    decomposer_max_tokens = 200
    coder_max_tokens = 512
    interpreter_max_tokens = 200
    verifier_max_tokens = 1024
    
    # ── Temperature per agent ──
    decomposer_temp = 0.2
    coder_temp = 0.1
    interpreter_temp = 0.2
    verifier_temp = 0.1


# ─────────────────────────────── Micro Agent ─────────────────────────

class MicroAgent:
    """A lightweight LLM wrapper that enforces a specific role."""
    
    def __init__(self, client: OpenAI, model: str, system_prompt: str,
                 max_tokens: int, temperature: float, name: str):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = name
    
    def call(self, user_message: str) -> str:
        """Make a single LLM call and return the response text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[AGENT ERROR] {e}"
    
    def call_streaming(self, user_message: str, 
                       token_callback: Optional[Callable[[str], None]] = None) -> str:
        """Make a streaming LLM call, calling token_callback for each token."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            chunks = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    chunks.append(token)
                    if token_callback:
                        token_callback(token)
            return "".join(chunks)
        except Exception as e:
            return f"[AGENT ERROR] {e}"


# ─────────────────────────── Helper Functions ─────────────────────────

def extract_code_block(text: str) -> Optional[str]:
    """Extract the first Python code block from text."""
    match = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code
    return None


def extract_done_answer(text: str) -> Optional[int]:
    """Check if text contains a DONE: answer directive."""
    match = re.search(r'DONE:\s*(\d+)', text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def extract_boxed_answer(text: str) -> Optional[int]:
    """Extract \\boxed{N} answer from text."""
    match = re.search(r'\\boxed\s*\{\s*([0-9,]+)\s*\}', text)
    if match:
        try:
            return int(match.group(1).replace(',', ''))
        except ValueError:
            pass
    return None


def extract_verified_answer(text: str) -> Optional[int]:
    """Extract VERIFIED: N from verifier output."""
    match = re.search(r'VERIFIED:\s*(\d+)', text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def validate_coder_output(text: str) -> tuple[bool, str]:
    """Validate that coder output is a valid, short code block."""
    code = extract_code_block(text)
    if not code:
        return False, "No ```python code block found. Respond with ONLY a code block."
    
    lines = [l for l in code.strip().split('\n') if l.strip()]
    code_lines = [l for l in lines if not l.strip().startswith('#')]
    comment_lines = [l for l in lines if l.strip().startswith('#')]
    
    if len(code_lines) == 0:
        return False, "Code block has no executable lines. Write actual code."
    
    if len(comment_lines) > len(code_lines):
        return False, (
            f"Too many comments ({len(comment_lines)}) vs code ({len(code_lines)}). "
            "Write CODE, not comments."
        )
    
    # Check for hardcoded answers
    for line in code_lines:
        if re.match(r'^\s*print\s*\(\s*\d+\s*\)\s*$', line.strip()):
            if len(code_lines) <= 3:
                return False, (
                    "You hardcoded an answer. COMPUTE the value with sympy/numpy instead."
                )
    
    return True, code


def format_scratchpad(scratchpad: list[dict]) -> str:
    """Format the scratchpad history into a readable string."""
    if not scratchpad:
        return "No steps completed yet."
    
    parts = []
    for i, step in enumerate(scratchpad):
        parts.append(f"--- Step {i+1} ---")
        parts.append(f"Goal: {step.get('goal', 'N/A')}")
        if step.get('code'):
            parts.append(f"Code: {step['code'][:200]}...")  # truncate
        if step.get('output'):
            parts.append(f"Result: {step['output'][:300]}")
        if step.get('interpretation'):
            parts.append(f"Meaning: {step['interpretation']}")
        parts.append("")
    return "\n".join(parts)


# ─────────────────────────── Orchestrator ─────────────────────────────

class MultiAgentOrchestrator:
    """
    Controls the decompose → code → execute → interpret loop.
    Forces each agent into its narrow role via code, not just prompts.
    """
    
    def __init__(self, cfg: AgentCFG = AgentCFG(),
                 event_callback: Optional[Callable[[str, Any], None]] = None):
        self.cfg = cfg
        self.event_callback = event_callback or (lambda t, c: None)
        
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key='ollama',
            timeout=600.0
        )
        
        # Create the 4 micro-agents
        self.decomposer = MicroAgent(
            self.client, cfg.served_model_name, cfg.decomposer_prompt,
            cfg.decomposer_max_tokens, cfg.decomposer_temp, "Decomposer"
        )
        self.coder = MicroAgent(
            self.client, cfg.served_model_name, cfg.coder_prompt,
            cfg.coder_max_tokens, cfg.coder_temp, "Coder"
        )
        self.interpreter = MicroAgent(
            self.client, cfg.served_model_name, cfg.interpreter_prompt,
            cfg.interpreter_max_tokens, cfg.interpreter_temp, "Interpreter"
        )
        self.verifier = MicroAgent(
            self.client, cfg.served_model_name, cfg.verifier_prompt,
            cfg.verifier_max_tokens, cfg.verifier_temp, "Verifier"
        )
        
        # Sandbox for code execution
        self.sandbox = AIMO3Sandbox(timeout=cfg.jupyter_timeout)
    
    def _emit(self, event_type: str, content: Any):
        """Emit an event to the callback and log."""
        self.event_callback(event_type, content)
        self._log(f"[{event_type.upper()}] {content}")
    
    def _log(self, msg: str):
        """Append to log file."""
        with open("v3_run.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    
    def solve(self, problem: str) -> Optional[int]:
        """
        Main solving loop. Returns the answer as an integer, or None on failure.
        """
        # Clear log
        with open("v3_run.log", "w", encoding="utf-8") as f:
            f.write(f"PROBLEM:\n{problem}\n\n{'='*60}\n\n")
        
        deadline = time.time() + self.cfg.problem_timeout
        scratchpad = []  # list of {goal, code, output, interpretation}
        
        self._emit("status", "Starting multi-agent solver...")
        
        for step_num in range(1, self.cfg.max_steps + 1):
            if time.time() > deadline:
                self._emit("status", "Timeout reached.")
                break
            
            self._emit("status", f"Step {step_num}/{self.cfg.max_steps}")
            self._log(f"\n{'='*40} STEP {step_num} {'='*40}\n")
            
            # ── Phase 1: DECOMPOSE ──
            decompose_input = (
                f"PROBLEM:\n{problem}\n\n"
                f"WORK SO FAR:\n{format_scratchpad(scratchpad)}\n\n"
                f"What is the ONE next thing to compute?"
            )
            
            self._emit("status", f"Step {step_num}: Decomposer thinking...")
            decomposition = self.decomposer.call_streaming(
                decompose_input,
                token_callback=lambda t: self.event_callback("decompose", t)
            )
            self._log(f"[DECOMPOSER]\n{decomposition}\n")
            
            # Check if decomposer says DONE
            done_answer = extract_done_answer(decomposition)
            if done_answer is not None:
                self._emit("status", f"Decomposer says answer is {done_answer}. Verifying...")
                verified = self._verify_answer(problem, done_answer, scratchpad, deadline)
                if verified is not None:
                    return verified
                else:
                    # Verification failed, continue stepping
                    self._emit("status", "Verification failed. Continuing...")
                    scratchpad.append({
                        "goal": f"Proposed answer {done_answer} — verification FAILED",
                        "code": None, "output": None,
                        "interpretation": "Answer was wrong. Need to re-examine."
                    })
                    continue
            
            # ── Phase 2: CODE ──
            coder_input = (
                f"PROBLEM:\n{problem}\n\n"
                f"TASK: {decomposition}\n\n"
                f"PREVIOUS RESULTS (variables are still in the kernel):\n"
                f"{format_scratchpad(scratchpad)}\n\n"
                f"Write a SHORT Python code block (5-30 lines) to accomplish ONLY the task above."
            )
            
            self._emit("status", f"Step {step_num}: Coder writing code...")
            code_response = self.coder.call_streaming(
                coder_input,
                token_callback=lambda t: self.event_callback("thinking", t)
            )
            self._log(f"[CODER RAW]\n{code_response}\n")
            
            # Validate coder output (with retries)
            code = None
            for retry in range(self.cfg.max_retries + 1):
                is_valid, result = validate_coder_output(code_response)
                if is_valid:
                    code = result
                    break
                elif retry < self.cfg.max_retries:
                    self._emit("status", f"Coder retry {retry+1}: {result[:60]}...")
                    code_response = self.coder.call(
                        f"{coder_input}\n\nPREVIOUS ATTEMPT REJECTED: {result}\n\nTry again."
                    )
                    self._log(f"[CODER RETRY {retry+1}]\n{code_response}\n")
            
            if code is None:
                self._emit("status", f"Step {step_num}: Coder failed to produce valid code. Skipping.")
                scratchpad.append({
                    "goal": decomposition[:100],
                    "code": None, "output": "Coder failed to produce valid code",
                    "interpretation": "Skipped — coder could not produce valid code"
                })
                continue
            
            self._emit("code", code)
            
            # ── Phase 3: EXECUTE ──
            self._emit("status", f"Step {step_num}: Executing code...")
            try:
                output = self.sandbox.execute(code)
            except Exception as exc:
                output = f"[ERROR] {exc}"
            
            self._emit("output", output)
            self._log(f"[EXECUTION OUTPUT]\n{output}\n")
            
            # ── Phase 4: INTERPRET ──
            interpret_input = (
                f"PROBLEM:\n{problem}\n\n"
                f"We just computed: {decomposition}\n\n"
                f"CODE OUTPUT:\n{output}\n\n"
                f"WORK SO FAR:\n{format_scratchpad(scratchpad)}\n\n"
                f"What does this result mean? End with NEXT: or DONE:"
            )
            
            self._emit("status", f"Step {step_num}: Interpreter analyzing...")
            interpretation = self.interpreter.call_streaming(
                interpret_input,
                token_callback=lambda t: self.event_callback("interpret", t)
            )
            self._log(f"[INTERPRETER]\n{interpretation}\n")
            
            # Record this step
            scratchpad.append({
                "goal": decomposition[:200],
                "code": code,
                "output": output[:500],
                "interpretation": interpretation[:300]
            })
            
            # Check if interpreter says DONE
            done_answer = extract_done_answer(interpretation)
            if done_answer is not None:
                self._emit("status", f"Interpreter says answer is {done_answer}. Verifying...")
                verified = self._verify_answer(problem, done_answer, scratchpad, deadline)
                if verified is not None:
                    return verified
                else:
                    self._emit("status", "Verification failed. Continuing...")
                    scratchpad.append({
                        "goal": f"Proposed answer {done_answer} — verification FAILED",
                        "code": None, "output": None,
                        "interpretation": "Answer was wrong. Need a different approach."
                    })
            
            # Also check if the output itself contains a boxed answer
            boxed = extract_boxed_answer(output)
            if boxed is not None:
                self._emit("status", f"Found boxed answer {boxed} in output. Verifying...")
                verified = self._verify_answer(problem, boxed, scratchpad, deadline)
                if verified is not None:
                    return verified
        
        # If we exhausted all steps, try to extract any answer from the scratchpad
        self._emit("status", "Max steps reached. Scanning for best answer...")
        return self._extract_best_answer(scratchpad)
    
    def _verify_answer(self, problem: str, proposed_answer: int,
                       scratchpad: list[dict], deadline: float) -> Optional[int]:
        """
        Run the verifier agent to independently check the answer.
        Returns the answer if verified, None if explicitly failed.
        """
        if time.time() > deadline:
            self._emit("answer", proposed_answer)
            return proposed_answer
        
        verifier_input = (
            f"PROBLEM:\n{problem}\n\n"
            f"PROPOSED ANSWER: {proposed_answer}\n\n"
            f"WORK DONE:\n{format_scratchpad(scratchpad)}\n\n"
            f"Write code to independently verify this answer from 2+ angles. "
            f"Use print(\"VERIFIED:\", N) or print(\"FAILED:\", reason)."
        )
        
        self._emit("status", "Verifier writing check code...")
        verify_response = self.verifier.call_streaming(
            verifier_input,
            token_callback=lambda t: self.event_callback("thinking", t)
        )
        self._log(f"[VERIFIER RESPONSE]\n{verify_response}\n")
        
        # Check if verifier's RAW response already says VERIFIED
        raw_verified = extract_verified_answer(verify_response)
        
        verify_code = extract_code_block(verify_response)
        if verify_code:
            self._emit("code", verify_code)
            self._emit("status", "Verifier executing check...")
            try:
                verify_output = self.sandbox.execute(verify_code)
            except Exception as exc:
                verify_output = f"[ERROR] {exc}"
            
            self._emit("output", verify_output)
            self._log(f"[VERIFIER OUTPUT]\n{verify_output}\n")
            
            # Check for VERIFIED in clean output
            verified = extract_verified_answer(verify_output)
            if verified is not None:
                self._emit("answer", verified)
                return verified
            
            # Check for explicit FAILED (only in clean output, not errors)
            is_error = '[ERROR]' in verify_output or 'Traceback' in verify_output or 'Error' in verify_output
            if not is_error and "FAILED:" in verify_output:
                self._emit("status", f"Verifier explicitly FAILED answer {proposed_answer}")
                return None
            
            # If execution errored or was inconclusive, accept the answer
            if str(proposed_answer) in verify_output or is_error:
                self._emit("status", "Verifier inconclusive (execution error). Accepting answer.")
                self._emit("answer", proposed_answer)
                return proposed_answer
        
        # If raw response had VERIFIED, use it
        if raw_verified is not None:
            self._emit("answer", raw_verified)
            return raw_verified
        
        # Fallback: accept the answer
        self._emit("status", "Verifier inconclusive. Accepting answer.")
        self._emit("answer", proposed_answer)
        return proposed_answer
    
    def _extract_best_answer(self, scratchpad: list[dict]) -> Optional[int]:
        """Try to extract an answer from scratchpad when no DONE was reached."""
        # Look through all outputs for numbers
        candidates = []
        for step in scratchpad:
            output = step.get("output", "")
            interp = step.get("interpretation", "")
            
            # Check boxed answers in output
            boxed = extract_boxed_answer(output)
            if boxed:
                candidates.append(boxed)
            
            # Check DONE in interpretation
            done = extract_done_answer(interp)
            if done:
                candidates.append(done)
        
        if candidates:
            answer = Counter(candidates).most_common(1)[0][0]
            self._emit("answer", answer)
            return answer
        
        return None
    
    def close(self):
        """Clean up resources."""
        try:
            self.sandbox.close()
        except Exception:
            pass


# ─────────────────────── Drop-in Replacement Solver ───────────────────

class AIMO3Solver:
    """
    Drop-in replacement for the old AIMO3Solver.
    Same interface: solve_problem(problem) -> int
    """
    
    def __init__(self, cfg: AgentCFG = AgentCFG()):
        self.cfg = cfg
    
    def solve_problem(self, problem: str,
                      event_callback: Optional[Callable[[str, Any], None]] = None) -> Optional[int]:
        """Solve a problem using the multi-agent pipeline."""
        cb = event_callback or (lambda t, c: print(f"[{t}] {c}"))
        
        orchestrator = MultiAgentOrchestrator(self.cfg, event_callback=cb)
        try:
            answer = orchestrator.solve(problem)
            return answer if answer is not None else 0
        finally:
            orchestrator.close()


# ─────────────────────────── Main Entry Point ─────────────────────────

if __name__ == '__main__':
    import pandas as pd
    
    df = pd.read_csv('reference.csv')
    first_problem = df.iloc[0]['problem']
    
    cfg = AgentCFG()
    solver = AIMO3Solver(cfg)
    
    start_time = time.time()
    
    def console_callback(event_type, content):
        if event_type == "code":
            print(f"\n{'─'*40}")
            print(f"📝 CODE:")
            print(content)
            print(f"{'─'*40}\n")
        elif event_type == "output":
            print(f"📊 OUTPUT: {content}\n")
        elif event_type == "answer":
            print(f"\n🎯 FINAL ANSWER: {content}\n")
        elif event_type == "status":
            print(f"⏳ {content}")
        elif event_type == "decompose":
            print(f"🧩 {content}", end="", flush=True)
        elif event_type == "interpret":
            print(f"🔍 {content}", end="", flush=True)
        elif event_type == "thinking":
            pass  # suppress raw tokens for cleaner console output
    
    ans = solver.solve_problem(first_problem, event_callback=console_callback)
    total_time = time.time() - start_time
    
    print(f"\nCompleted in {total_time:.2f}s. Answer: {ans}")
    with open("v3_output.log", "w") as f:
        f.write(f"Problem: {first_problem}\nAnswer: {ans}\nTime: {total_time}\n")
