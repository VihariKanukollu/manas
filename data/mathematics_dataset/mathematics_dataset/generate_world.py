"""
Generate math problems in "world model" format: step-by-step solution traces.

Instead of just (question, answer) pairs, we output:
    - equation: the original equation as a string
    - steps: list of algebraic operations and their results
    - final_answer: the solution

Output is JSONL format, one problem per line.

Usage:
    python -m mathematics_dataset.generate_world --num_train=10000 --num_test=1000 --output_dir=./world_data
"""

import os
import json
import random
import argparse

import sympy
from sympy import Symbol, Eq

from mathematics_dataset.sample import linear_system
from mathematics_dataset.sample import number
from mathematics_dataset.world_solver import equation_to_trace, format_trace_as_text


def generate_linear_1d(entropy_range=(3, 10)):
    """
    Generate a random 1D linear equation and its step-by-step solution.
    Returns a dict with equation, steps, final_answer.
    """
    entropy = random.uniform(*entropy_range)
    variable = Symbol('x')

    # generate a random solution
    solution = number.integer(entropy / 2, signed=True)

    # generate an equation with that solution
    equations = linear_system.linear_system(
        variables=[variable],
        solutions=[solution],
        entropy=entropy,
    )
    equation = equations[0]  # linear_system returns a list, we just want one

    # convert ops.Eq to sympy Eq using the .sympy() method
    eq = equation.sympy()

    # get the step-by-step trace
    trace = equation_to_trace(eq, variable)

    return {
        "equation": str(eq),
        "steps": trace["steps"],
        "final_answer": trace["final_answer"],
        "ground_truth": str(solution),  # for verification
    }


def generate_linear_2d(entropy_range=(3, 10)):
    """
    Generate a random 2D linear system (2 equations, 2 variables).
    For now, we just solve for x and return that trace.
    """
    entropy = random.uniform(*entropy_range)
    x, y = Symbol('x'), Symbol('y')

    # generate random solutions
    solution_x = number.integer(entropy / 4, signed=True)
    solution_y = number.integer(entropy / 4, signed=True)

    # generate equations with those solutions
    equations = linear_system.linear_system(
        variables=[x, y],
        solutions=[solution_x, solution_y],
        entropy=entropy,
    )

    # for now, just return the system as a string and the solution
    # TODO: implement step-by-step for systems (substitution, elimination, etc.)
    eq_strs = [str(eq) for eq in equations]

    return {
        "equation": " and ".join(eq_strs),
        "steps": [{"op": "solve system", "result": f"x = {solution_x}, y = {solution_y}"}],
        "final_answer": f"x = {solution_x}, y = {solution_y}",
        "ground_truth_x": str(solution_x),
        "ground_truth_y": str(solution_y),
    }


def generate_dataset(num_examples, entropy_range, problem_type="linear_1d"):
    """Generate a list of problems."""
    problems = []
    generators = {
        "linear_1d": generate_linear_1d,
        "linear_2d": generate_linear_2d,
    }
    generator = generators[problem_type]

    for _ in range(num_examples):
        try:
            problem = generator(entropy_range)
            problems.append(problem)
        except Exception as e:
            # some random equations might fail, just skip
            print(f"Warning: skipped problem due to {e}")
            continue

    return problems


def format_for_training(problem):
    """
    Convert a problem dict to a training string.
    This is the format the model will be trained on.
    """
    lines = []
    lines.append(f"Solve: {problem['equation']}")
    for i, step in enumerate(problem['steps'], 1):
        lines.append(f"Step {i}: {step['op']} -> {step['result']}")
    lines.append(f"Answer: {problem['final_answer']}")
    return "\n".join(lines)


def save_jsonl(problems, filepath):
    """Save problems to a JSONL file."""
    with open(filepath, 'w') as f:
        for problem in problems:
            # add the formatted training string
            problem['text'] = format_for_training(problem)
            f.write(json.dumps(problem) + '\n')
    print(f"Saved {len(problems)} problems to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate math world-model training data")
    parser.add_argument("--num_train", type=int, default=10000, help="Number of training examples")
    parser.add_argument("--num_test_id", type=int, default=1000, help="Number of in-distribution test examples")
    parser.add_argument("--num_test_ood", type=int, default=1000, help="Number of OOD test examples")
    parser.add_argument("--output_dir", type=str, default="./world_data", help="Output directory")
    parser.add_argument("--problem_type", type=str, default="linear_1d", choices=["linear_1d", "linear_2d"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Training data: medium entropy (medium difficulty)
    print(f"Generating {args.num_train} training examples...")
    train_problems = generate_dataset(args.num_train, entropy_range=(3, 8), problem_type=args.problem_type)
    save_jsonl(train_problems, os.path.join(args.output_dir, "train.jsonl"))

    # In-distribution test: same entropy range as training
    print(f"Generating {args.num_test_id} in-distribution test examples...")
    test_id_problems = generate_dataset(args.num_test_id, entropy_range=(3, 8), problem_type=args.problem_type)
    save_jsonl(test_id_problems, os.path.join(args.output_dir, "test_id.jsonl"))

    # Out-of-distribution test: higher entropy (harder/larger numbers)
    print(f"Generating {args.num_test_ood} OOD test examples...")
    test_ood_problems = generate_dataset(args.num_test_ood, entropy_range=(10, 15), problem_type=args.problem_type)
    save_jsonl(test_ood_problems, os.path.join(args.output_dir, "test_ood.jsonl"))

    print("Done!")

    # Print a few examples
    print("\n--- Sample training examples ---")
    for problem in train_problems[:3]:
        print(problem['text'])
        print()


if __name__ == "__main__":
    main()

