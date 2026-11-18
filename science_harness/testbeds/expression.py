"""A bounded arithmetic expression interpreter, never eval/exec of model code."""
import ast
import math
import operator


FUNCTIONS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log,
             "sin": math.sin, "cos": math.cos, "abs": abs}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv}


class Expression:
    def __init__(self, expression):
        if not isinstance(expression, str) or not 1 <= len(expression) <= 1000:
            raise ValueError("expression must contain 1..1000 characters")
        self.root = ast.parse(expression, mode="eval").body
        if len(list(ast.walk(self.root))) > 128:
            raise ValueError("Expression too complex")
        self._validate(self.root)

    def _validate(self, node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            raise ValueError("Use ** for exponentiation, not ^; bitwise operations are not supported")
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not math.isfinite(node.value):
                raise ValueError("Only finite numeric literals are allowed")
        elif isinstance(node, ast.Name) and node.id in ("x", "pi", "e"):
            pass
        elif isinstance(node, ast.BinOp) and type(node.op) in (*OPERATORS, ast.Pow):
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._validate(node.operand)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
            self._validate(node.args[0])
        else:
            raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")

    def __call__(self, x):
        return self._compute(self.root, x)

    def _compute(self, node, x):
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name):
            value = {"x": x, "pi": math.pi, "e": math.e}[node.id]
        elif isinstance(node, ast.UnaryOp):
            value = self._compute(node.operand, x)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.Call):
            value = FUNCTIONS[node.func.id](self._compute(node.args[0], x))
        else:
            left, right = self._compute(node.left, x), self._compute(node.right, x)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 16:
                    raise ValueError("Exponent magnitude exceeds 16")
                value = float(left) ** float(right)
            else:
                value = OPERATORS[type(node.op)](left, right)
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Expression produced a nonfinite or excessive result")
        return float(value)
