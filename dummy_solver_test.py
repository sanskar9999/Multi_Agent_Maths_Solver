
import time
from v2_solver_final import AIMO3Solver, CFG

class DummyCFG(CFG):
    workers = 1
    attempts = 1
    turns = 3
    base_problem_timeout = 600

if __name__ == '__main__':
    problem = "Calculate 123 + 456. Output the final answer in \\boxed{}."
    
    cfg = DummyCFG()
    solver = AIMO3Solver(cfg)
    
    start_time = time.time()
    try:
        ans = solver.solve_problem(problem)
        total_time = time.time() - start_time
        print(f"Dummy test completed in {total_time:.2f}s. Result: {ans}")
    except Exception as e:
        print(f"Error during dummy test: {e}")
    finally:
        # Cleanup kernels if needed, though solver should handle it
        pass
