import os
import pandas as pd
from openai import OpenAI
from v2_solver import AIMO3Solver, CFG

class TestCFG(CFG):
    attempts = 1
    turns = 1
    base_problem_timeout = 60

if __name__ == '__main__':
    cfg = TestCFG()
    solver = AIMO3Solver(cfg)
    
    problem = "What is 2 + 2? Return the answer in \\boxed{}."
    print(f"\nTesting with simple problem: {problem}")
    
    ans = solver.solve_problem(problem)
    print(f"Test result: {ans}")
