"""
Solve equations step-by-step and return the solution trace.

This is the core "world model" piece: instead of just returning the final answer,
we expose the intermediate algebraic states as the model learns to predict.

Example:
    equation: 2*x + 5 = 17
    steps:
        1. subtract 5 from both sides -> 2*x = 12
        2. divide both sides by 2 -> x = 6
    final_answer: 6
"""

import sympy
from sympy import Symbol, Eq, solve, simplify
from sympy.core.numbers import Integer, Rational


def solve_linear_1d_with_steps(equation, variable):
    """
    Solve a linear equation in one variable, returning step-by-step trace.

    Args:
        equation: A sympy Eq object, e.g. Eq(2*x + 5, 17)
        variable: The sympy Symbol to solve for, e.g. Symbol('x')

    Returns:
        dict with keys:
            - equation: string of the original equation
            - steps: list of dicts, each with 'op', 'result'
            - final_answer: the solution value (as string)
            - valid: bool, whether we successfully solved it
    """
    steps = []
    lhs = equation.lhs
    rhs = equation.rhs

    # we'll iteratively isolate the variable
    # strategy: move constant terms to rhs, then divide by coefficient

    # Step 1: collect terms with variable on lhs, constants on rhs
    # expand and simplify first
    lhs = sympy.expand(lhs)
    rhs = sympy.expand(rhs)

    # get the constant term on lhs (terms without the variable)
    lhs_const = lhs.as_independent(variable)[0]
    lhs_var = lhs - lhs_const  # the part with the variable

    # get the variable term on rhs (if any)
    rhs_const = rhs.as_independent(variable)[0]
    rhs_var = rhs - rhs_const

    # move variable terms to lhs, constants to rhs
    new_lhs = lhs_var - rhs_var
    new_rhs = rhs_const - lhs_const

    new_lhs = simplify(new_lhs)
    new_rhs = simplify(new_rhs)

    if new_lhs != lhs or new_rhs != rhs:
        steps.append({
            "op": "collect terms",
            "result": f"{new_lhs} = {new_rhs}"
        })

    # Step 2: divide by the coefficient of the variable
    coeff = new_lhs.as_coefficient(variable)
    if coeff is None:
        # might be just the variable itself
        coeff = 1 if new_lhs == variable else None

    if coeff is None or coeff == 0:
        # something went wrong, fall back to sympy solve
        solutions = solve(equation, variable)
        if solutions:
            return {
                "equation": str(equation),
                "steps": [{"op": "solve directly", "result": f"{variable} = {solutions[0]}"}],
                "final_answer": str(solutions[0]),
                "valid": True
            }
        else:
            return {
                "equation": str(equation),
                "steps": [],
                "final_answer": None,
                "valid": False
            }

    if coeff != 1:
        final_value = simplify(new_rhs / coeff)
        steps.append({
            "op": f"divide by {coeff}",
            "result": f"{variable} = {final_value}"
        })
    else:
        final_value = new_rhs

    return {
        "equation": str(equation),
        "steps": steps,
        "final_answer": str(final_value),
        "valid": True
    }


def solve_quadratic_with_steps(equation, variable):
    """
    Solve a quadratic equation, returning step-by-step trace.
    For now, just uses sympy's solve and wraps it.
    TODO: implement proper step-by-step (factoring, quadratic formula, etc.)
    """
    solutions = solve(equation, variable)
    if not solutions:
        return {
            "equation": str(equation),
            "steps": [],
            "final_answer": None,
            "valid": False
        }

    # format multiple solutions
    if len(solutions) == 1:
        answer_str = str(solutions[0])
    else:
        # sort by real part for nice ordering, but don't fail on complex
        def sort_key(s):
            if s.is_real:
                return (0, float(s))
            return (1, str(s))  # non-real solutions go last
        answer_str = ", ".join(str(s) for s in sorted(solutions, key=sort_key))

    return {
        "equation": str(equation),
        "steps": [{"op": "solve", "result": f"{variable} = {answer_str}"}],
        "final_answer": answer_str,
        "valid": True
    }


def equation_to_trace(equation, variable):
    """
    Main entry point. Given an equation, return the solution trace.
    Automatically dispatches to the right solver based on equation degree.
    """
    # figure out the degree
    lhs_expanded = sympy.expand(equation.lhs - equation.rhs)
    poly = sympy.Poly(lhs_expanded, variable)
    degree = poly.degree()

    if degree == 1:
        return solve_linear_1d_with_steps(equation, variable)
    elif degree == 2:
        return solve_quadratic_with_steps(equation, variable)
    else:
        # fallback: just solve it
        solutions = solve(equation, variable)
        if solutions:
            answer_str = ", ".join(str(s) for s in solutions)
            return {
                "equation": str(equation),
                "steps": [{"op": "solve", "result": f"{variable} = {answer_str}"}],
                "final_answer": answer_str,
                "valid": True
            }
        return {
            "equation": str(equation),
            "steps": [],
            "final_answer": None,
            "valid": False
        }


def format_trace_as_text(trace):
    """
    Convert a solution trace dict to a human-readable training string.
    This is what we'll feed to the model.
    """
    lines = []
    lines.append(f"Equation: {trace['equation']}")
    for i, step in enumerate(trace['steps'], 1):
        lines.append(f"Step {i}: {step['op']} -> {step['result']}")
    if trace['final_answer']:
        lines.append(f"Final answer: {trace['final_answer']}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Quick test
if __name__ == "__main__":
    x = Symbol('x')

    # Test 1: simple linear
    eq1 = Eq(2*x + 5, 17)
    trace1 = equation_to_trace(eq1, x)
    print("Test 1: 2x + 5 = 17")
    print(format_trace_as_text(trace1))
    print()

    # Test 2: linear with variable on both sides
    eq2 = Eq(3*x + 4, x + 10)
    trace2 = equation_to_trace(eq2, x)
    print("Test 2: 3x + 4 = x + 10")
    print(format_trace_as_text(trace2))
    print()

    # Test 3: quadratic
    eq3 = Eq(x**2 - 5*x + 6, 0)
    trace3 = equation_to_trace(eq3, x)
    print("Test 3: x^2 - 5x + 6 = 0")
    print(format_trace_as_text(trace3))
    print()

    # Test 4: simple
    eq4 = Eq(x, 5)
    trace4 = equation_to_trace(eq4, x)
    print("Test 4: x = 5")
    print(format_trace_as_text(trace4))

