import random
import re
from dataclasses import dataclass, field
from typing import Callable

from base import Data, Env, Verifier


OPERATOR_SYMBOLS = ["⊕", "⊗", "⊖", "⊘", "⊙", "⊛", "⊜", "⊞", "⊠", "⊡"]

OperatorFactory = Callable[[list[int]], tuple[str, Callable[[int, int], int]]]

OPERATOR_TEMPLATES: list[OperatorFactory] = [
    lambda c: (
        f"{c[0]}*a + {c[1]}*b + {c[2]}",
        lambda a, b, _c=c: _c[0] * a + _c[1] * b + _c[2],
    ),
    lambda c: (
        f"{c[0]}*a - {c[1]}*b",
        lambda a, b, _c=c: _c[0] * a - _c[1] * b,
    ),
    lambda c: (
        f"{c[0]}*a*b + {c[1]}",
        lambda a, b, _c=c: _c[0] * a * b + _c[1],
    ),
    lambda c: (
        f"a^2 + {c[0]}*b + {c[1]}",
        lambda a, b, _c=c: a**2 + _c[0] * b + _c[1],
    ),
    lambda c: (
        f"{c[0]}*a^2 - b",
        lambda a, b, _c=c: _c[0] * a**2 - b,
    ),
    lambda c: (
        f"{c[0]}*(a^2 + b^2) + {c[1]}",
        lambda a, b, _c=c: _c[0] * (a**2 + b**2) + _c[1],
    ),
    lambda c: (
        f"{c[0]}*(a - b)^2",
        lambda a, b, _c=c: _c[0] * (a - b) ** 2,
    ),
    lambda c: (
        f"{c[0]}*a + {c[1]}*b - {c[2]}",
        lambda a, b, _c=c: _c[0] * a + _c[1] * b - _c[2],
    ),
]

TEMPLATE_NUM_COEFFS = [3, 2, 2, 2, 1, 2, 1, 3]


@dataclass
class Operator:
    symbol: str
    formula_str: str
    compute: Callable[[int, int], int]

    def definition_str(self) -> str:
        return f"a {self.symbol} b = {self.formula_str}"


def sample_operator(symbol: str, rng: random.Random, difficulty: int = 1) -> Operator:
    if difficulty == 1:
        template_pool = list(range(2))
        coeff_range = (1, 3)
    elif difficulty == 2:
        template_pool = list(range(4))
        coeff_range = (1, 4)
    else:
        template_pool = list(range(len(OPERATOR_TEMPLATES)))
        coeff_range = (1, 5)

    idx = rng.choice(template_pool)
    n_coeffs = TEMPLATE_NUM_COEFFS[idx]
    coeffs = [rng.randint(*coeff_range) for _ in range(n_coeffs)]
    formula_str, compute_fn = OPERATOR_TEMPLATES[idx](coeffs)
    return Operator(symbol=symbol, formula_str=formula_str, compute=compute_fn)


def sample_operator_with_meta(
    symbol: str, rng: random.Random, difficulty: int = 1
) -> tuple[Operator, int, list[int]]:
    if difficulty == 1:
        template_pool = list(range(2))
        coeff_range = (1, 3)
    elif difficulty == 2:
        template_pool = list(range(4))
        coeff_range = (1, 4)
    else:
        template_pool = list(range(len(OPERATOR_TEMPLATES)))
        coeff_range = (1, 5)

    idx = rng.choice(template_pool)
    n_coeffs = TEMPLATE_NUM_COEFFS[idx]
    coeffs = [rng.randint(*coeff_range) for _ in range(n_coeffs)]
    formula_str, compute_fn = OPERATOR_TEMPLATES[idx](coeffs)
    op = Operator(symbol=symbol, formula_str=formula_str, compute=compute_fn)
    return op, idx, coeffs


def reconstruct_operator(symbol: str, template_idx: int, coeffs: list[int]) -> Operator:
    formula_str, compute_fn = OPERATOR_TEMPLATES[template_idx](coeffs)
    return Operator(symbol=symbol, formula_str=formula_str, compute=compute_fn)


@dataclass
class ExprNode:
    value: int | None = None
    operator: Operator | None = None
    left: "ExprNode | None" = None
    right: "ExprNode | None" = None

    def is_leaf(self) -> bool:
        return self.value is not None

    def evaluate(self) -> int:
        if self.is_leaf():
            return self.value
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        return self.operator.compute(left_val, right_val)

    def to_str(self) -> str:
        if self.is_leaf():
            return str(self.value)
        left_str = self.left.to_str()
        right_str = self.right.to_str()
        if not self.left.is_leaf():
            left_str = f"({left_str})"
        if not self.right.is_leaf():
            right_str = f"({right_str})"
        return f"{left_str} {self.operator.symbol} {right_str}"

    def get_evaluation_steps(self) -> list[dict]:
        if self.is_leaf():
            return []

        steps = []
        steps.extend(self.left.get_evaluation_steps())
        steps.extend(self.right.get_evaluation_steps())

        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        result = self.operator.compute(left_val, right_val)

        steps.append(
            {
                "expr_str": f"{left_val} {self.operator.symbol} {right_val}",
                "left_val": left_val,
                "right_val": right_val,
                "operator": self.operator,
                "result": result,
            }
        )
        return steps


def sample_expression(
    operators: list[Operator],
    rng: random.Random,
    depth: int = 2,
    operand_range: tuple[int, int] = (1, 15),
) -> ExprNode:
    if depth <= 0:
        return ExprNode(value=rng.randint(*operand_range))

    op = rng.choice(operators)

    shape = rng.choice(["balanced", "left", "right"])
    if shape == "balanced" and depth >= 2:
        left = sample_expression(operators, rng, depth - 1, operand_range)
        right = sample_expression(operators, rng, depth - 1, operand_range)
    elif shape == "left":
        left = sample_expression(operators, rng, depth - 1, operand_range)
        right = ExprNode(value=rng.randint(*operand_range))
    else:
        left = ExprNode(value=rng.randint(*operand_range))
        right = sample_expression(operators, rng, depth - 1, operand_range)

    return ExprNode(operator=op, left=left, right=right)


def serialize_expr(node: ExprNode) -> dict:
    if node.is_leaf():
        return {"type": "leaf", "value": node.value}
    return {
        "type": "op",
        "symbol": node.operator.symbol,
        "left": serialize_expr(node.left),
        "right": serialize_expr(node.right),
    }


def deserialize_expr(d: dict, op_lookup: dict[str, Operator]) -> ExprNode:
    if d["type"] == "leaf":
        return ExprNode(value=d["value"])
    return ExprNode(
        operator=op_lookup[d["symbol"]],
        left=deserialize_expr(d["left"], op_lookup),
        right=deserialize_expr(d["right"], op_lookup),
    )


STEP_PATTERN = re.compile(
    r"(-?\d+)\s*([⊕⊗⊖⊘⊙⊛⊜⊞⊠⊡])\s*(-?\d+)\s*=\s*(?:.*=\s*)?(-?\d+)"
)

EXPLICIT_ANSWER_PATTERN = re.compile(
    r"(?:answer|result|final)\s*(?:is|:)\s*(-?\d+)", re.IGNORECASE
)
TRAILING_EQUALS_PATTERN = re.compile(r"=\s*(-?\d+)\s*$", re.MULTILINE)


@dataclass
class ParsedStep:
    left_val: int
    op_symbol: str
    right_val: int
    claimed_result: int


def parse_cot(response: str) -> tuple[list[ParsedStep], int | None]:
    steps = []
    for match in STEP_PATTERN.finditer(response):
        steps.append(
            ParsedStep(
                left_val=int(match.group(1)),
                op_symbol=match.group(2),
                right_val=int(match.group(3)),
                claimed_result=int(match.group(4)),
            )
        )

    final_answer = None
    explicit = EXPLICIT_ANSWER_PATTERN.search(response)
    if explicit:
        final_answer = int(explicit.group(1))
    else:
        trailing = list(TRAILING_EQUALS_PATTERN.finditer(response))
        if trailing:
            final_answer = int(trailing[-1].group(1))
        elif steps:
            final_answer = steps[-1].claimed_result

    return steps, final_answer


@dataclass
class RewardInfo:
    total_reward: float
    step_reward: float
    final_reward: float
    n_correct_steps: int
    n_parsed_steps: int
    n_expected_steps: int
    final_answer_correct: bool
    step_details: list[dict] = field(default_factory=list)


def compute_reward(
    response: str,
    expression: ExprNode,
    operators: list[Operator],
    step_weight: float = 0.5,
    format_penalty: float = 0.1,
) -> RewardInfo:
    expected_steps = expression.get_evaluation_steps()
    expected_final = expression.evaluate()

    parsed_steps, parsed_final = parse_cot(response)

    op_lookup: dict[str, Operator] = {op.symbol: op for op in operators}

    n_correct = 0
    step_details = []
    for ps in parsed_steps:
        op = op_lookup.get(ps.op_symbol)
        if op is None:
            step_details.append(
                {"parsed": ps, "correct": False, "reason": "unknown operator"}
            )
            continue
        true_result = op.compute(ps.left_val, ps.right_val)
        is_correct = ps.claimed_result == true_result
        if is_correct:
            n_correct += 1
        step_details.append(
            {
                "claimed": ps.claimed_result,
                "true": true_result,
                "correct": is_correct,
                "expr": f"{ps.left_val} {ps.op_symbol} {ps.right_val}",
            }
        )

    n_expected = len(expected_steps)
    if parsed_steps:
        step_reward = n_correct / max(len(parsed_steps), n_expected)
    else:
        step_reward = -format_penalty

    final_correct = parsed_final is not None and parsed_final == expected_final
    final_reward = 1.0 if final_correct else 0.0

    total = step_weight * step_reward + (1.0 - step_weight) * final_reward
    total = max(total, 0.0)

    return RewardInfo(
        total_reward=total,
        step_reward=step_reward,
        final_reward=final_reward,
        n_correct_steps=n_correct,
        n_parsed_steps=len(parsed_steps),
        n_expected_steps=n_expected,
        final_answer_correct=final_correct,
        step_details=step_details,
    )


class NovelOpsVerifier(Verifier):
    def extract_answer(self, test_solution: str) -> str:
        _, final_answer = parse_cot(test_solution)
        return str(final_answer) if final_answer is not None else ""

    def verify(self, data: Data, test_answer: str) -> bool:
        return test_answer.strip() == data.answer.strip()

    def compute_detailed_reward(
        self,
        data: Data,
        test_solution: str,
        step_weight: float = 0.5,
        format_penalty: float = 0.1,
    ) -> RewardInfo:
        operators = self._reconstruct_operators(data.metadata)
        op_lookup = {op.symbol: op for op in operators}
        expression = deserialize_expr(data.metadata["expression_tree"], op_lookup)
        return compute_reward(
            test_solution,
            expression,
            operators,
            step_weight=step_weight,
            format_penalty=format_penalty,
        )

    @staticmethod
    def _reconstruct_operators(metadata: dict) -> list[Operator]:
        return [
            reconstruct_operator(om["symbol"], om["template_idx"], om["coeffs"])
            for om in metadata["operators"]
        ]


class NovelOpsEnv(Env):
    def __init__(self, seed: int | None = None):
        super().__init__(name="novel_ops", verifier=NovelOpsVerifier)
        self.rng = random.Random(seed)

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: int = 1,
    ) -> list[Data]:
        results = []
        attempts = 0
        limit = max_attempts * num_of_questions
        while len(results) < num_of_questions and attempts < limit:
            attempts += 1
            data = self._generate_one(difficulty)
            if data is not None:
                results.append(data)
        return results

    def extract_answer(self, test_solution: str) -> str:
        return self.verifier.extract_answer(test_solution)

    def _generate_one(self, difficulty: int) -> Data | None:
        n_operators = min((difficulty - 1) // 3 + 1, 3)
        expression_depth = min((difficulty - 1) // 3 + 1, 3)
        operator_difficulty = min((difficulty - 1) // 3 + 1, 3)
        operand_range = (1, 10 + difficulty)

        symbols = self.rng.sample(OPERATOR_SYMBOLS, n_operators)
        operators_meta = [
            sample_operator_with_meta(s, self.rng, operator_difficulty) for s in symbols
        ]
        operators = [om[0] for om in operators_meta]

        expr = sample_expression(operators, self.rng, expression_depth, operand_range)

        try:
            answer = expr.evaluate()
            if abs(answer) > 100_000:
                return None
        except (OverflowError, RecursionError):
            return None

        steps = expr.get_evaluation_steps()
        expr_str = expr.to_str()

        definitions = "\n".join(op.definition_str() for op in operators)
        prompt = (
            f"Below are definitions of new mathematical operators.\n\n"
            f"Definitions:\n{definitions}\n\n"
            f"Compute step by step: {expr_str}\n\n"
            f"Show each intermediate computation on its own line in the format:\n"
            f"<operand> <operator> <operand> = <expansion> = <result>\n"
            f"Then write your final answer as: Answer: <number>"
        )

        metadata = {
            "operators": [
                {"symbol": om[0].symbol, "template_idx": om[1], "coeffs": om[2]}
                for om in operators_meta
            ],
            "expression_tree": serialize_expr(expr),
            "expected_steps": [
                {"expr_str": s["expr_str"], "result": s["result"]} for s in steps
            ],
        }

        return Data(
            question=prompt,
            answer=str(answer),
            difficulty=difficulty,
            metadata=metadata,
        )
