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
        'You must use Python code to solve the problem. '
        'Whenever you need to calculate something, write a Python code block starting with ```python and ending with ```. '
        'The code will be executed in a stateful environment and the output will be provided to you. '
        'You must use print() to output results from your code. '
        'The final answer must be a single non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}. '
    )
    
    preference_prompt = (
        'You have access to `math`, `numpy` and `sympy` to solve the problem.'
    )

    served_model_name = 'svjack/Qwen3-4B-Instruct-2507-heretic'
    base_url = 'http://127.0.0.1:11434/v1'
    
    high_problem_timeout = 900
    base_problem_timeout = 270

    jupyter_timeout = 30.0
    sandbox_timeout = 3

    context_tokens = 4096
    buffer_tokens = 512
    search_tokens = 50
    early_stop = 2 # Reduced for faster local testing
    attempts = 4 # Multiple attempts for voting
    workers = 1 # Single worker for local Ollama to avoid memory swapping and interleaved logs
    turns = 10
    seed = 42
    
    temperature = 0.8 # Slightly lower temperature for more deterministic math

class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

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
            'import numpy\n'
            'import sympy\n'
            'import mpmath\n'
            'import itertools\n'
            'import collections\n'
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

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr
        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

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
            'import numpy\n'
            'import sympy\n'
            'import mpmath\n'
            'import itertools\n'
            'import collections\n'
            'mpmath.mp.dps = 64\n'
        )

    def __del__(self):
        self.close()

class AIMO3Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=self.cfg.base_url, api_key='ollama', timeout=600.0)
        self._initialize_kernels()
        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _initialize_kernels(self) -> None:
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()
        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')

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
                
        pattern = r'final\s+answer\s+is\s*([0-9,]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
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
        
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem}
            ]

            for turn in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                stream = self.client.chat.completions.create(
                    model=self.cfg.served_model_name,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    seed=attempt_seed,
                    max_tokens=self.cfg.buffer_tokens,
                    stream=True
                )
                
                content_chunks = []
                print(f"Attempt {attempt_index} generating response: ", end="", flush=True)
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
                
                # Check for answer
                answer = self._scan_for_answer(content)
                if answer is not None:
                    final_answer = answer
                    break
                    
                # Check for code block to execute
                code_pattern = r'```(?:python)?(.*?)```'
                code_matches = re.findall(code_pattern, content, re.DOTALL)
                
                if code_matches:
                    code_to_run = code_matches[-1] # Run the last code block
                    python_calls += 1
                    print(f"Attempt {attempt_index} generating code:\n{code_to_run.strip()}")
                    
                    try:
                        output = sandbox.execute(code_to_run)
                    except TimeoutError as exc:
                        output = f'[ERROR] {exc}'
                        
                    if '[ERROR]' in output or 'Traceback' in output:
                        python_errors += 1
                    
                    print(f"Attempt {attempt_index} execution output:\n{output.strip()}\n")
                    
                    messages.append({
                        "role": "user", 
                        "content": f"```output\n{output}\n```\nAnalyze the output and continue."
                    })
                else:
                    # If no code block and no answer, ask it to continue or provide answer
                    messages.append({
                        "role": "user",
                        "content": "Please continue reasoning or provide the final answer in \\boxed{}."
                    })

        except Exception as exc:
            python_errors += 1
            print(f"Error in attempt {attempt_index}: {exc}")
            with open("exception.log", "a") as f:
                traceback.print_exc(file=f)

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

    def _select_answer(self, detailed_results: list) -> int:
        answer_votes = defaultdict(int)

        for result in detailed_results:
            answer = result['Answer']
            if answer is not None:
                answer_votes[answer] += 1

        if not answer_votes:
            print('\nFinal Answer: 0\n')
            return 0

        # Most common
        final_answer = max(answer_votes.items(), key=lambda x: x[1])[0]
        print(f'\nFinal Answer: {final_answer} (Votes: {answer_votes[final_answer]})\n')

        return final_answer
    
    def solve_problem(self, problem: str) -> int:
        print(f'\nProblem: {problem}\n')
        user_input = f'{problem} {self.cfg.preference_prompt}'
        
        budget = 300 # Simplify budget for testing
        deadline = time.time() + budget

        tasks = []
        for attempt_index in range(self.cfg.attempts):
            tasks.append((self.cfg.system_prompt, attempt_index))

        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()

        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

        try:
            futures = []
            for (system_prompt, attempt_index) in tasks:
                future = executor.submit(
                    self._process_attempt, 
                    user_input, 
                    system_prompt, 
                    attempt_index, 
                    stop_event, 
                    deadline
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)

                    if result['Answer'] is not None:
                        valid_answers.append(result['Answer'])

                    counts = Counter(valid_answers).most_common(1)
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break

                except Exception as exc:
                    print(f'Future failed: {exc}')
                    continue

        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)

        if detailed_results:
            results_dataframe = pd.DataFrame(detailed_results)
            print(results_dataframe)

        if not valid_answers:
            print('\nResult: 0\n')
            return 0

        return self._select_answer(detailed_results)

    def __del__(self):
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
                except Exception:
                    pass

if __name__ == '__main__':
    # Test with first problem from reference.csv
    df = pd.read_csv('reference.csv')
    first_problem = df.iloc[0]['problem']
    
    cfgs = CFG()
    solver = AIMO3Solver(cfgs)
    
    print("\nStarting test on first problem...")
    solver.solve_problem(first_problem)
