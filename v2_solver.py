import os
import re
import math
import time
import queue
import threading
import contextlib
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from jupyter_client import KernelManager

class CFG:
    system_prompt = (
        'You are an expert mathematical problem solver. '
        'You must solve the problem using Python code. '
        'CRITICAL RULE 1: The very first part of your response must be a Python code block (```python). '
        'CRITICAL RULE 2: DO NOT perform manual calculations or "mental math" in comments. '
        'CRITICAL RULE 3: Every mathematical step, including intermediate values and condition checks, MUST be performed by Python code. '
        'Solve the problem step-by-step using `sympy` or coordinate geometry. '
        'The code will be executed in a stateful environment. '
        'You must use print() to output results. '
        'The final answer must be a single non-negative integer between 0 and 99999, placed in \\boxed{}.'
    )
    
    preference_prompt = (
        'START YOUR RESPONSE WITH A PYTHON CODE BLOCK. '
        'Use `sympy` to model the geometry. Define points, lines, and circles. '
        'Use coordinate geometry to solve for the unknown conditions. '
        'Do not assume known results; verify them programmatically. '
        'Iterate through candidate integer side lengths (a, b, c) and use code to verify if the geometric condition (e.g., Y lies on AD) holds for each.'
    )

    served_model_name = 'svjack/Qwen3-4B-Instruct-2507-heretic'
    base_url = 'http://127.0.0.1:11434/v1'
    
    high_problem_timeout = 900
    base_problem_timeout = 600 # Increased for difficult problems

    jupyter_timeout = 60.0
    sandbox_timeout = 5

    context_tokens = 8192 # Increased for longer reasoning
    buffer_tokens = 2048 # Significantly increased to prevent code clipping
    search_tokens = 50
    early_stop = 1
    attempts = 1 # Balanced for speed and depth
    workers = 1 # Single worker for local Ollama to avoid memory swapping
    turns = 8 # More turns for depth
    seed = 42
    
    # temperatures for different attempts
    temperatures = [0.1, 0.1, 0.7]

class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50100

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import numpy as np\n'
            'import sympy as sp\n'
            'import mpmath\n'
            'import itertools\n'
            'import collections\n'
            'from sympy import symbols, Eq, solve, sqrt, Plane, Point3D, Line3D, Circle\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        # Auto-add print wrapper if last line is an expression
        lines = code.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith('#') and 'print' not in last_line and 'import ' not in last_line and '=' not in last_line:
                lines[-1] = f'print({last_line})'
                code = '\n'.join(lines)

        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])
                stderr_parts.append(self._format_error(traceback_list))
            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')
            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts).strip()
        stderr = ''.join(stderr_parts).strip()

        if stderr:
            return f'STDOUT:\n{stdout}\nSTDERR:\n{stderr}' if stdout else stderr
        return stdout if stdout else '[WARN] No output. Use print() to see results.'

    def close(self):
        try:
            if self._client:
                self._client.stop_channels()
        except Exception:
            pass
        if self._owns_kernel and self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
            try:
                self._km.cleanup_resources()
            except Exception:
                pass

    def reset(self):
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy as np\n'
            'import sympy as sp\n'
            'from sympy import symbols, Eq, solve, sqrt\n'
        )

    def __del__(self):
        self.close()

class AIMO3Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=self.cfg.base_url, api_key='ollama', timeout=600.0)
        self._initialize_kernels()

    def _initialize_kernels(self) -> None:
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        self.sandbox_pool = queue.Queue()
        def _create_sandbox():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())
        print('Kernels initialized.\n')

    def _scan_for_answer(self, text: str) -> int | None:
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)
        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass
        return None

    def _process_attempt(self, problem: str, system_prompt: str, attempt_index: int, stop_event: threading.Event, deadline: float) -> dict:
        if stop_event.is_set() or time.time() > deadline:
            return {'Attempt': attempt_index + 1, 'Answer': None}

        sandbox = None
        python_calls = 0
        python_errors = 0
        final_answer = None
        
        # Use specific temperature from config
        temp = self.cfg.temperatures[attempt_index] if attempt_index < len(self.cfg.temperatures) else 0.7
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{problem}\n\n{self.cfg.preference_prompt}"}
            ]

            for turn in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                with open("v2_full_run.log", "a", encoding="utf-8") as debug_f:
                    debug_f.write(f"\n[Attempt {attempt_index} Turn {turn}] Starting API call...\n")
                
                try:
                    stream = self.client.chat.completions.create(
                        model=self.cfg.served_model_name,
                        messages=messages,
                        temperature=temp,
                        seed=attempt_seed,
                        max_tokens=self.cfg.buffer_tokens,
                        stream=True
                    )
                except Exception as api_exc:
                    with open("v2_full_run.log", "a", encoding="utf-8") as debug_f:
                        debug_f.write(f"\n[Attempt {attempt_index} Turn {turn}] API CALL FAILED: {api_exc}\n")
                    break

                content_chunks = []
                print(f"[Attempt {attempt_index} Turn {turn}] Thinking: ", end="", flush=True)
                for chunk in stream:
                    if stop_event.is_set() or time.time() > deadline:
                        break
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        print(token, end="", flush=True)
                        content_chunks.append(token)
                print("\n")
                content = "".join(content_chunks)
                messages.append({"role": "assistant", "content": content})
                
                with open("v2_full_run.log", "a", encoding="utf-8") as debug_f:
                    debug_f.write(f"\n[Attempt {attempt_index} Turn {turn}] Assistant:\n{content}\n")
                
                # Check for answer
                answer = self._scan_for_answer(content)
                if answer is not None:
                    final_answer = answer
                    break
                    
                # Check for code block to execute
                code_pattern = r'```(?:python)?(.*?)```'
                code_matches = re.findall(code_pattern, content, re.DOTALL)
                
                if code_matches:
                    code_to_run = code_matches[-1].strip() # Run the last code block
                    python_calls += 1
                    print(f"[Attempt {attempt_index}] Executing Code...")
                    
                    try:
                        output = sandbox.execute(code_to_run)
                    except Exception as exc:
                        output = f'[ERROR] {exc}'
                        
                    if '[ERROR]' in output or 'Traceback' in output or 'STDERR' in output:
                        python_errors += 1
                    
                    print(f"[Attempt {attempt_index}] Output:\n{output}\n")
                    
                    with open("v2_full_run.log", "a", encoding="utf-8") as debug_f:
                        debug_f.write(f"\n[Attempt {attempt_index} Turn {turn}] Code Output:\n{output}\n")
                    
                    messages.append({
                        "role": "user", 
                        "content": f"```output\n{output}\n```\nAnalyze the output and proceed."
                    })
                else:
                    # If no code block in Turn 0/1, force it
                    if turn <= 1:
                        force_prompt = "You haven't used Python yet. Please use `sympy` to model this problem mathematically and solve it step-by-step."
                    else:
                        force_prompt = "Please continue or provide the final answer in \\boxed{}."
                    
                    messages.append({
                        "role": "user",
                        "content": force_prompt
                    })

        except Exception as exc:
            python_errors += 1
            print(f"Error in attempt {attempt_index}: {exc}")

        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        return {
            'Attempt': attempt_index + 1, 
            'Python Calls': python_calls, 
            'Python Errors': python_errors, 
            'Answer': final_answer
        }

    def solve_problem(self, problem: str) -> int:
        print(f'\nSOLVING: {problem}\n')
        with open("v2_full_run.log", "w", encoding="utf-8") as debug_log:
            debug_log.write(f"SOLVING: {problem}\n\n")
            
        deadline = time.time() + self.cfg.base_problem_timeout
        tasks = [(self.cfg.system_prompt, i) for i in range(self.cfg.attempts)]
        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()

        # Wrap process_attempt to log to file
        original_process = self._process_attempt
        def logged_process(*args, **kwargs):
            res = original_process(*args, **kwargs)
            with open("v2_full_run.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- ATTEMPT {args[2]} RESULT ---\n{res}\n")
            return res

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(logged_process, problem, sp, idx, stop_event, deadline) for (sp, idx) in tasks]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    detailed_results.append(res)
                    if res['Answer'] is not None:
                        valid_answers.append(res['Answer'])
                        counts = Counter(valid_answers).most_common(1)
                        if counts and counts[0][1] >= self.cfg.early_stop:
                            stop_event.set()
                            break
                except Exception as exc:
                    print(f'Future error: {exc}')

        if not valid_answers:
            return 0
        
        answer_votes = Counter(valid_answers)
        final_answer = answer_votes.most_common(1)[0][0]
        print(f'\nFinal Answer Selection: {final_answer} (Votes: {answer_votes[final_answer]})\n')
        return final_answer

if __name__ == '__main__':
    df = pd.read_csv('reference.csv')
    first_problem = df.iloc[0]['problem']
    
    cfg = CFG()
    solver = AIMO3Solver(cfg)
    
    start_time = time.time()
    ans = solver.solve_problem(first_problem)
    total_time = time.time() - start_time
    
    print(f"Test completed in {total_time:.2f}s. Result: {ans}")
    with open("v2_output.log", "w") as f:
        f.write(f"Problem: {first_problem}\nAnswer: {ans}\nTime: {total_time}\n")
