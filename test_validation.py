
import re

def _validate_code(code: str) -> tuple[bool, str]:
    """Validate that code is actual computation, not comment-heavy fake code."""
    lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
    if not lines:
        return False, "Empty code block."
    comment_lines = sum(1 for l in lines if l.startswith('#'))
    code_lines = sum(1 for l in lines if not l.startswith('#'))
    if code_lines == 0:
        return False, "Code block contains ONLY comments and no executable code."
    
    executable = [l for l in lines if not l.startswith('#')]
    for line in executable:
        match = re.match(r'^print\s*\(\s*(\d+)\s*\)$', line)
        if match and len(executable) <= 3:
            return False, (
                f"REJECTED: You hardcoded the answer as print({match.group(1)}). "
                "The answer must be COMPUTED by the code, not guessed. "
                "Write code that uses sympy/numpy to solve the problem."
            )
    return True, ""

# Test cases
test_cases = [
    {
        "name": "Heavy Comments (Should now pass)",
        "code": """
# This is a comment
# Another comment
# More comments
# Even more
# So many comments
x = 1 + 1
print(x)
""",
        "expected": True
    },
    {
        "name": "Only Comments (Should still fail)",
        "code": """
# Just comments
# No code here
""",
        "expected": False
    },
    {
        "name": "Hardcoded (Should still fail)",
        "code": """
print(123)
""",
        "expected": False
    },
    {
        "name": "Normal Code (Should pass)",
        "code": """
import sympy as sp
x = sp.symbols('x')
print(sp.solve(x - 1, x))
""",
        "expected": True
    }
]

for tc in test_cases:
    valid, msg = _validate_code(tc['code'])
    print(f"Test: {tc['name']}")
    print(f"Valid: {valid}")
    if msg:
        print(f"Message: {msg}")
    assert valid == tc['expected']
    print("-" * 20)

print("All tests passed!")
