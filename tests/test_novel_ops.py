import json
import random

import pytest

from base import Data
from novel_ops import (
    OPERATOR_TEMPLATES,
    TEMPLATE_NUM_COEFFS,
    ExprNode,
    NovelOpsEnv,
    NovelOpsVerifier,
    Operator,
    ParsedStep,
    RewardInfo,
    compute_reward,
    deserialize_expr,
    parse_cot,
    reconstruct_operator,
    sample_expression,
    sample_operator,
    sample_operator_with_meta,
    serialize_expr,
)


class TestOperatorTemplates:
    @pytest.mark.parametrize("idx", range(len(OPERATOR_TEMPLATES)))
    def test_each_template_produces_callable(self, idx):
        n = TEMPLATE_NUM_COEFFS[idx]
        coeffs = [2] * n
        formula_str, fn = OPERATOR_TEMPLATES[idx](coeffs)
        assert isinstance(formula_str, str)
        assert callable(fn)
        result = fn(3, 4)
        assert isinstance(result, int)

    def test_linear_template(self):
        _, fn = OPERATOR_TEMPLATES[0]([2, 3, 1])
        assert fn(5, 7) == 2 * 5 + 3 * 7 + 1

    def test_asymmetric_template(self):
        _, fn = OPERATOR_TEMPLATES[1]([3, 2])
        assert fn(10, 4) == 3 * 10 - 2 * 4

    def test_product_template(self):
        _, fn = OPERATOR_TEMPLATES[2]([2, 5])
        assert fn(3, 4) == 2 * 3 * 4 + 5

    def test_square_template(self):
        _, fn = OPERATOR_TEMPLATES[3]([3, 1])
        assert fn(4, 2) == 4**2 + 3 * 2 + 1

    def test_mixed_square_template(self):
        _, fn = OPERATOR_TEMPLATES[4]([2])
        assert fn(5, 3) == 2 * 5**2 - 3

    def test_sum_of_squares_template(self):
        _, fn = OPERATOR_TEMPLATES[5]([2, 1])
        assert fn(3, 4) == 2 * (9 + 16) + 1

    def test_difference_template(self):
        _, fn = OPERATOR_TEMPLATES[6]([3])
        assert fn(7, 2) == 3 * (7 - 2) ** 2

    def test_last_linear_template(self):
        _, fn = OPERATOR_TEMPLATES[7]([2, 3, 4])
        assert fn(5, 6) == 2 * 5 + 3 * 6 - 4


class TestSampleOperator:
    def test_returns_operator(self):
        rng = random.Random(42)
        op = sample_operator("⊕", rng, difficulty=1)
        assert isinstance(op, Operator)
        assert op.symbol == "⊕"

    def test_difficulty_1_only_linear(self):
        rng = random.Random(0)
        for _ in range(50):
            op = sample_operator("⊕", rng, difficulty=1)
            assert "a*b" not in op.formula_str
            assert "^2" not in op.formula_str

    def test_definition_str(self):
        rng = random.Random(42)
        op = sample_operator("⊗", rng)
        defn = op.definition_str()
        assert "⊗" in defn
        assert defn.startswith("a ⊗ b = ")


class TestSampleOperatorWithMeta:
    def test_returns_tuple(self):
        rng = random.Random(42)
        result = sample_operator_with_meta("⊕", rng)
        assert len(result) == 3
        op, idx, coeffs = result
        assert isinstance(op, Operator)
        assert isinstance(idx, int)
        assert isinstance(coeffs, list)

    def test_reconstruct_matches(self):
        rng = random.Random(42)
        op, idx, coeffs = sample_operator_with_meta("⊕", rng)
        rebuilt = reconstruct_operator("⊕", idx, coeffs)
        assert rebuilt.formula_str == op.formula_str
        for a, b in [(1, 2), (5, 10), (0, 0), (-3, 7)]:
            assert rebuilt.compute(a, b) == op.compute(a, b)


class TestExprNode:
    def test_leaf(self):
        node = ExprNode(value=42)
        assert node.is_leaf()
        assert node.evaluate() == 42
        assert node.to_str() == "42"
        assert node.get_evaluation_steps() == []

    def test_single_op(self):
        _, fn = OPERATOR_TEMPLATES[0]([1, 1, 0])
        op = Operator(symbol="⊕", formula_str="1*a + 1*b + 0", compute=fn)
        node = ExprNode(
            operator=op,
            left=ExprNode(value=3),
            right=ExprNode(value=5),
        )
        assert not node.is_leaf()
        assert node.evaluate() == 8
        assert node.to_str() == "3 ⊕ 5"

    def test_nested_expression(self):
        _, fn = OPERATOR_TEMPLATES[0]([1, 1, 0])
        op = Operator(symbol="⊕", formula_str="1*a + 1*b + 0", compute=fn)
        inner = ExprNode(operator=op, left=ExprNode(value=2), right=ExprNode(value=3))
        outer = ExprNode(operator=op, left=inner, right=ExprNode(value=4))
        assert outer.evaluate() == (2 + 3) + 4
        assert outer.to_str() == "(2 ⊕ 3) ⊕ 4"

    def test_evaluation_steps_order(self):
        _, fn = OPERATOR_TEMPLATES[1]([1, 1])
        op = Operator(symbol="⊖", formula_str="1*a - 1*b", compute=fn)
        left = ExprNode(operator=op, left=ExprNode(value=10), right=ExprNode(value=3))
        root = ExprNode(operator=op, left=left, right=ExprNode(value=2))
        steps = root.get_evaluation_steps()
        assert len(steps) == 2
        assert steps[0]["result"] == 7
        assert steps[1]["result"] == 5


class TestSampleExpression:
    def test_depth_0_is_leaf(self):
        rng = random.Random(42)
        op = sample_operator("⊕", rng)
        node = sample_expression([op], rng, depth=0)
        assert node.is_leaf()

    def test_depth_1_has_one_step(self):
        rng = random.Random(42)
        op = sample_operator("⊕", rng)
        node = sample_expression([op], rng, depth=1)
        steps = node.get_evaluation_steps()
        assert len(steps) == 1

    def test_depth_2_has_multiple_steps(self):
        rng = random.Random(42)
        op = sample_operator("⊕", rng)
        node = sample_expression([op], rng, depth=2)
        steps = node.get_evaluation_steps()
        assert len(steps) >= 2


class TestSerializeDeserialize:
    def test_leaf_roundtrip(self):
        node = ExprNode(value=7)
        d = serialize_expr(node)
        assert d == {"type": "leaf", "value": 7}
        rebuilt = deserialize_expr(d, {})
        assert rebuilt.is_leaf()
        assert rebuilt.evaluate() == 7

    def test_tree_roundtrip(self):
        rng = random.Random(42)
        op, idx, coeffs = sample_operator_with_meta("⊕", rng)
        expr = sample_expression([op], rng, depth=2)
        original_result = expr.evaluate()

        d = serialize_expr(expr)
        rebuilt_op = reconstruct_operator("⊕", idx, coeffs)
        rebuilt = deserialize_expr(d, {"⊕": rebuilt_op})
        assert rebuilt.evaluate() == original_result

    def test_serialized_is_json_safe(self):
        rng = random.Random(42)
        op = sample_operator("⊕", rng)
        expr = sample_expression([op], rng, depth=2)
        d = serialize_expr(expr)
        json_str = json.dumps(d)
        assert json.loads(json_str) == d


class TestParseCot:
    def test_single_step_with_answer(self):
        response = "3 ⊕ 5 = 2*3 + 5 = 11\nAnswer: 11"
        steps, final = parse_cot(response)
        assert len(steps) == 1
        assert steps[0].left_val == 3
        assert steps[0].op_symbol == "⊕"
        assert steps[0].right_val == 5
        assert steps[0].claimed_result == 11
        assert final == 11

    def test_multi_step(self):
        response = "4 ⊗ 3 = 24\n24 ⊕ 2 = 50\nAnswer: 50"
        steps, final = parse_cot(response)
        assert len(steps) == 2
        assert final == 50

    def test_explicit_answer_pattern(self):
        response = "Some work...\nThe answer is 42"
        _, final = parse_cot(response)
        assert final == 42

    def test_result_keyword(self):
        response = "result: 99"
        _, final = parse_cot(response)
        assert final == 99

    def test_trailing_equals_fallback(self):
        response = "computation goes here\n= 77"
        _, final = parse_cot(response)
        assert final == 77

    def test_last_step_fallback(self):
        response = "5 ⊕ 3 = 13"
        steps, final = parse_cot(response)
        assert final == 13
        assert len(steps) == 1

    def test_negative_numbers(self):
        response = "-3 ⊕ 5 = -1\nAnswer: -1"
        steps, final = parse_cot(response)
        assert steps[0].left_val == -3
        assert steps[0].claimed_result == -1
        assert final == -1

    def test_no_parseable_content(self):
        response = "I don't know how to solve this."
        steps, final = parse_cot(response)
        assert steps == []
        assert final is None

    def test_multiple_equals_in_expansion(self):
        # expansion has two = signs: formula = intermediate = result
        response = "3 ⊕ 5 = 2*3 + 1*5 + 0 = 6 + 5 + 0 = 11\nAnswer: 11"
        steps, final = parse_cot(response)
        assert len(steps) == 1
        assert steps[0].claimed_result == 11
        assert final == 11

    def test_three_equals_in_expansion(self):
        # even more intermediate steps on one line
        response = "8 ⊖ 14 = 4*8 - 3*14 = 32 - 42 = -10\nAnswer: -10"
        steps, final = parse_cot(response)
        assert len(steps) == 1
        assert steps[0].left_val == 8
        assert steps[0].right_val == 14
        assert steps[0].claimed_result == -10
        assert final == -10

    def test_multi_step_with_expansions(self):
        # each step has expansion = intermediate = result
        response = (
            "8 ⊖ 14 = 4*8 - 3*14 = 32 - 42 = -10\n"
            "4 ⊖ -10 = 4*4 - 3*(-10) = 16 + 30 = 46\n"
            "-10 ⊜ 46 = 1*(-10)*46 + 1 = -460 + 1 = -459\n"
            "Answer: -459"
        )
        steps, final = parse_cot(response)
        assert len(steps) == 3
        assert steps[0].claimed_result == -10
        assert steps[1].claimed_result == 46
        assert steps[2].claimed_result == -459
        assert final == -459

    def test_expansion_starts_with_digit(self):
        # expansion like "2*3 + 5" starts with a digit — parser must not grab it
        response = "3 ⊕ 5 = 2*3 + 5 = 11\nAnswer: 11"
        steps, final = parse_cot(response)
        assert steps[0].claimed_result == 11

    def test_parenthesized_negative_in_expansion(self):
        # negative operand in expansion: 3*(-5)
        response = "4 ⊖ -10 = 4*4 - 3*(-10) = 16 + 30 = 46\nAnswer: 46"
        steps, final = parse_cot(response)
        assert len(steps) == 1
        assert steps[0].claimed_result == 46


class TestComputeReward:
    def _make_simple_setup(self):
        _, fn = OPERATOR_TEMPLATES[0]([2, 1, 0])
        op = Operator(symbol="⊕", formula_str="2*a + 1*b + 0", compute=fn)
        expr = ExprNode(operator=op, left=ExprNode(value=3), right=ExprNode(value=5))
        return expr, [op]

    def test_perfect_response(self):
        expr, ops = self._make_simple_setup()
        expected = expr.evaluate()
        response = f"3 ⊕ 5 = 2*3 + 1*5 + 0 = {expected}\nAnswer: {expected}"
        reward = compute_reward(response, expr, ops)
        assert reward.total_reward == 1.0
        assert reward.final_answer_correct
        assert reward.n_correct_steps == 1

    def test_wrong_final_answer(self):
        expr, ops = self._make_simple_setup()
        response = "3 ⊕ 5 = 11\nAnswer: 9999"
        reward = compute_reward(response, expr, ops)
        assert not reward.final_answer_correct
        assert reward.final_reward == 0.0

    def test_wrong_step(self):
        expr, ops = self._make_simple_setup()
        expected = expr.evaluate()
        response = f"3 ⊕ 5 = 999\nAnswer: {expected}"
        reward = compute_reward(response, expr, ops)
        assert reward.n_correct_steps == 0
        assert reward.final_answer_correct

    def test_no_steps_format_penalty(self):
        expr, ops = self._make_simple_setup()
        response = "I think the answer is 11"
        reward = compute_reward(response, expr, ops, format_penalty=0.2)
        assert reward.step_reward == -0.2

    def test_reward_clipped_at_zero(self):
        expr, ops = self._make_simple_setup()
        response = "no idea"
        reward = compute_reward(response, expr, ops)
        assert reward.total_reward >= 0.0

    def test_step_weight_affects_total(self):
        expr, ops = self._make_simple_setup()
        expected = expr.evaluate()
        response = f"3 ⊕ 5 = {expected}\nAnswer: {expected}"
        r1 = compute_reward(response, expr, ops, step_weight=0.0)
        r2 = compute_reward(response, expr, ops, step_weight=1.0)
        assert r1.total_reward == pytest.approx(1.0)
        assert r2.total_reward == pytest.approx(1.0)

        wrong_step_response = f"3 ⊕ 5 = 9999\nAnswer: {expected}"
        r3 = compute_reward(wrong_step_response, expr, ops, step_weight=0.0)
        r4 = compute_reward(wrong_step_response, expr, ops, step_weight=1.0)
        assert r3.total_reward > r4.total_reward


class TestNovelOpsVerifier:
    def test_extract_answer_explicit(self):
        v = NovelOpsVerifier()
        assert v.extract_answer("Answer: 42") == "42"

    def test_extract_answer_negative(self):
        v = NovelOpsVerifier()
        assert v.extract_answer("Answer: -7") == "-7"

    def test_extract_answer_empty(self):
        v = NovelOpsVerifier()
        assert v.extract_answer("no numbers here") == ""

    def test_verify_correct(self):
        v = NovelOpsVerifier()
        data = Data(question="q", answer="42")
        assert v.verify(data, "42")

    def test_verify_wrong(self):
        v = NovelOpsVerifier()
        data = Data(question="q", answer="42")
        assert not v.verify(data, "99")

    def test_verify_whitespace(self):
        v = NovelOpsVerifier()
        data = Data(question="q", answer="42")
        assert v.verify(data, " 42 ")

    def test_compute_detailed_reward(self):
        env = NovelOpsEnv(seed=42)
        episodes = env.generate(num_of_questions=1, difficulty=1)
        ep = episodes[0]

        ops = NovelOpsVerifier._reconstruct_operators(ep.metadata)
        op_lookup = {op.symbol: op for op in ops}
        expr = deserialize_expr(ep.metadata["expression_tree"], op_lookup)
        steps = expr.get_evaluation_steps()

        lines = []
        for s in steps:
            lines.append(
                f"{s['left_val']} {s['operator'].symbol} {s['right_val']} = {s['result']}"
            )
        lines.append(f"Answer: {ep.answer}")
        response = "\n".join(lines)

        reward = env.verifier.compute_detailed_reward(ep, response)
        assert reward.total_reward == 1.0
        assert reward.final_answer_correct


class TestNovelOpsEnv:
    def test_init(self):
        env = NovelOpsEnv(seed=42)
        assert env.name == "novel_ops"
        assert isinstance(env.verifier, NovelOpsVerifier)

    def test_generate_returns_data_list(self):
        env = NovelOpsEnv(seed=42)
        episodes = env.generate(num_of_questions=5, difficulty=1)
        assert len(episodes) == 5
        for ep in episodes:
            assert isinstance(ep, Data)

    def test_generate_data_fields(self):
        env = NovelOpsEnv(seed=42)
        ep = env.generate(num_of_questions=1, difficulty=3)[0]
        assert isinstance(ep.question, str)
        assert isinstance(ep.answer, str)
        assert ep.difficulty == 3
        assert "operators" in ep.metadata
        assert "expression_tree" in ep.metadata
        assert "expected_steps" in ep.metadata

    def test_question_contains_definitions(self):
        env = NovelOpsEnv(seed=42)
        ep = env.generate(num_of_questions=1, difficulty=1)[0]
        assert "Definitions:" in ep.question
        assert "Compute step by step:" in ep.question

    def test_answer_bounded(self):
        env = NovelOpsEnv(seed=42)
        episodes = env.generate(num_of_questions=100, difficulty=8)
        for ep in episodes:
            assert abs(int(ep.answer)) <= 100_000

    def test_difficulty_scales_complexity(self):
        env_easy = NovelOpsEnv(seed=42)
        env_hard = NovelOpsEnv(seed=42)
        easy_eps = env_easy.generate(num_of_questions=20, difficulty=1)
        hard_eps = env_hard.generate(num_of_questions=20, difficulty=10)
        easy_steps = sum(len(e.metadata["expected_steps"]) for e in easy_eps)
        hard_steps = sum(len(e.metadata["expected_steps"]) for e in hard_eps)
        assert hard_steps > easy_steps

    def test_difficulty_1_single_operator(self):
        env = NovelOpsEnv(seed=42)
        ep = env.generate(num_of_questions=1, difficulty=1)[0]
        assert len(ep.metadata["operators"]) == 1

    def test_difficulty_10_multiple_operators(self):
        env = NovelOpsEnv(seed=42)
        ep = env.generate(num_of_questions=1, difficulty=10)[0]
        assert len(ep.metadata["operators"]) == 3

    def test_seed_reproducibility(self):
        ep1 = NovelOpsEnv(seed=99).generate(num_of_questions=5, difficulty=5)
        ep2 = NovelOpsEnv(seed=99).generate(num_of_questions=5, difficulty=5)
        for a, b in zip(ep1, ep2):
            assert a.question == b.question
            assert a.answer == b.answer

    def test_extract_answer(self):
        env = NovelOpsEnv(seed=42)
        assert env.extract_answer("Answer: 123") == "123"

    def test_verify_integration(self):
        env = NovelOpsEnv(seed=42)
        ep = env.generate(num_of_questions=1, difficulty=1)[0]
        assert env.verify(ep, ep.answer)
        assert not env.verify(ep, "99999999")

    def test_metadata_json_serializable(self):
        env = NovelOpsEnv(seed=42)
        ep = env.generate(num_of_questions=1, difficulty=5)[0]
        json_str = json.dumps(ep.metadata)
        assert json.loads(json_str) == ep.metadata

    def test_data_to_json_roundtrip(self):
        env = NovelOpsEnv(seed=42)
        ep = env.generate(num_of_questions=1, difficulty=5)[0]
        json_str = ep.to_json_str()
        restored = Data.from_json_str(json_str)
        assert restored.question == ep.question
        assert restored.answer == ep.answer
        assert restored.difficulty == ep.difficulty
        assert restored.metadata == ep.metadata

    def test_generate_respects_max_attempts(self):
        env = NovelOpsEnv(seed=42)
        episodes = env.generate(num_of_questions=1000, max_attempts=1, difficulty=1)
        assert len(episodes) <= 1000
