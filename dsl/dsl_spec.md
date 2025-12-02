# Gyan DSL Specification (v1)

## 1. Goal

Define a *domain-agnostic* reasoning DSL that is expressive enough to
reconstruct the semantic operations used in DeepMind's
`mathematics_dataset`, while being small and compositional.

This spec is derived programmatically from:

    python -m dev.analyze_math_dsl

which scans:

    data/mathematics_dataset/mathematics_dataset/**

and lists all calls/attribute uses into:

    sympy, number, ops, polynomials, probability,
    combinatorics, composition

We then group these into DSL primitive families.

## 2. High-level Types

We keep types intentionally coarse; many modules share the same shapes.

* `Int`       – machine integer
* `Rat`       – rational number (SymPy `Rational`)
* `Num`       – `Int | Rat`
* `Var`       – variable symbol (e.g. `"x"`)
* `Expr`      – symbolic expression in one or more variables
* `Poly`      – polynomial in one variable (specialized `Expr`)
* `Equation`  – `Expr == Expr`
* `Ineq`      – `Expr < Expr`, `Expr <= Expr`, `Expr > Expr`, `Expr >= Expr`
* `Bool`      – boolean
* `Seq[T]`    – ordered sequence (tuple)
* `Set[T]`    – finite set
* `Event`     – probability event on a finite space

## 3. Math DSL Primitives

Below, *right column* shows which analyzed functions are covered.

### 3.1 Numeric Core

These are the basic building blocks for arithmetic and number theory.

* `add(a: Num, b: Num) -> Num`
    - covers: `ops.Add`, `sympy.Add`
* `sub(a: Num, b: Num) -> Num`
    - covers: `ops.Sub`
* `mul(a: Num, b: Num) -> Num`
    - covers: `ops.Mul`, `sympy.Mul`
* `div(a: Num, b: Num) -> Num`
    - covers: `ops.Div`
* `pow(a: Num, k: Int) -> Num`
    - covers: `ops.Pow`, `sympy.Pow`
* `sqrt(a: Num) -> Num`
    - covers: `ops.Sqrt`, `sympy.sqrt`
* `neg(a: Num) -> Num`
    - covers: `ops.Neg`
* `gcd(a: Int, b: Int) -> Int`
    - covers: `sympy.gcd`
* `lcm(a: Int, b: Int) -> Int`
    - covers: `sympy.lcm`
* `factorint(n: Int) -> Dict[Int, Int]`
    - covers: `sympy.factorint`
* `factorial(n: Int) -> Int`
    - covers: `sympy.factorial`
* `prod(seq: Seq[Num]) -> Num`
    - covers: `sympy.prod`
* `is_integer(x) -> Bool`
    - covers: `number.is_integer`, `number.is_integer_or_rational`,
      `number.is_integer_or_rational_or_decimal`

Note: functions like `number.integer`, `integer_or_decimal`, etc. are
data *sampling* utilities, not semantic reasoning steps, so they do not
need one-to-one DSL primitives.

### 3.2 Expression / Polynomial

We abstract over SymPy's polynomial utilities.

* `var(name: str) -> Var`
    - covers: `sympy.Symbol`, `sympy.symbols`, `sympy.var`
* `const(value: Num) -> Expr`
    - covers: `ops.Constant`, `sympy.Integer`, `sympy.Rational`
* `make_poly(coeffs: Seq[Num], var: Var) -> Poly`
    - covers: `sympy.Poly`, `polynomials.coefficients_to_polynomial`
* `eval_expr(expr: Expr, env: Dict[Var, Num]) -> Num`
    - covers: `sympy.sympify`, evaluation in many modules
* `expand_expr(expr: Expr) -> Expr`
    - covers: `sympy.expand`, `polynomials.expand_coefficients`
* `factor_expr(expr: Expr) -> Expr`
    - covers: `sympy.factor`
* `simplify_expr(expr: Expr) -> Expr`
    - covers: `sympy.simplify`
* `differentiate(expr: Expr, var: Var, order: Int = 1) -> Expr`
    - covers: `polynomials.differentiate`, calculus module
* `integrate_poly(expr: Poly, var: Var) -> Poly`
    - covers: `polynomials.integrate`

The remaining polynomial helper functions in the generator
(`sample_coefficients`, `sample_with_brackets`, etc.) are used to
produce *questions*, not for student reasoning, so the DSL need not
expose them directly.

### 3.3 Equations and Inequalities

These are central for algebra. They connect arithmetic/polynomials with
comparison tasks.

* `eq(lhs: Expr, rhs: Expr) -> Equation`
    - covers: `ops.Eq`, `sympy.Eq`
* `ineq(op: str, lhs: Expr, rhs: Expr) -> Ineq`
    - covers: comparison operators: `<, <=, >, >=, ==, !=` and related
      symbols (`EQ_SYMBOL`, `NE_SYMBOL`, `GE_SYMBOL`, `GT_SYMBOL`,
      `LE_SYMBOL`, `LT_SYMBOL`)
* `add_both_sides(e: Equation, delta: Expr) -> Equation`
* `mul_both_sides(e: Equation, factor: Expr) -> Equation`
* `div_both_sides(e: Equation, factor: Expr) -> Equation`
* `substitute(e: Equation, var: Var, repl: Expr) -> Equation`
* `is_solution(e: Equation, var: Var, value: Num) -> Bool`
    - covers: algebra `linear_1d`, `linear_2d`, `polynomial_roots`
      semantics (checking that a proposed root satisfies the equation).

We can treat polynomial root finding itself as *search over sequences of
these steps* rather than a primitive.

### 3.4 Sets, Sequences, and Ordering

Most of this already exists in the imported DSL (containers, `size`,
`merge`, `argmax`, etc.) but we surface a few key reasoning ops.

* `sort(seq: Seq[Num]) -> Seq[Num]`
    - covers: comparison `sort`, ordering tasks
* `kth_largest(seq: Seq[Num], k: Int) -> Num`
    - covers: `kth_biggest`
* `closest_to(seq: Seq[Num], target: Num) -> Num`
    - covers: `closest`

### 3.5 Probability & Combinatorics

Finite-event constructs for probability modules:

* `finite_product_event(events: Seq[Event]) -> Event`
    - covers: `FiniteProductEvent`
* `count_level_set(counts: Dict[Outcome, Int]) -> Event`
    - covers: `CountLevelSetEvent`
* `event_prob(event: Event) -> Rat`
    - derived from `probability` utilities in `util/probability.py`
* `uniform_positive_ints_with_sum(n: Int, total: Int) -> Seq[Int]`
    - covers: `uniform_positive_integers_with_sum`
* `uniform_non_negative_ints_with_sum(n: Int, total: Int) -> Seq[Int]`
    - covers: `uniform_non_negative_integers_with_sum`

### 3.6 Coverage Notes

* All `sympy`/`ops` arithmetic and comparison operators used in the
  dataset map onto the numeric and equation families above.
* Polynomial manipulation (`differentiate`, `integrate`, `expand`,
  `coefficients_to_polynomial`, etc.) is representable via the
  expression/polynomial family.
* Number-theoretic functions (`gcd`, `lcm`, `factorint`, `nextprime`,
  `randprime`) are covered by numeric primitives; `nextprime`/`randprime`
  are only used for sampling and do not need direct DSL counterparts.
* Dataset-internal abstractions (`Context`, `Entity`, `FunctionHandle`,
  `Polynomial` wrappers) are not surfaced in the DSL; they are part of
  the question generator, not the reasoning model.

This spec should be treated as the *minimal required surface*: any DSL
we implement must at least be able to express these families to ensure
that all math tasks in `mathematics_dataset` can, in principle, be
represented as sequences of DSL transformations.

## 4. ARC-style DSL (Existing)

In addition to the math-derived spec above, Gyan already ships with a
large, hand-crafted DSL in `dsl/dsl.py` (ported from
`michaelhodel/arc-dsl` but now treated as the generic Gyan DSL).

Static analysis shows:

* **160 top-level DSL functions** are currently defined in `dsl/dsl.py`.
* They fall roughly into these families:

### 4.1 Numeric / Logic / Container Utilities
     - `identity`, `add`, `subtract`, `multiply`, `divide`, `invert`,
       `even`, `double`, `halve`, `flip`, `equality`, `contained`,
       `combine`, `intersection`, `difference`, `dedupe`, `order`,
       `repeat`, `greater`, `size`, `merge`, `maximum`, `minimum`,
       `valmax`, `valmin`, `argmax`, `argmin`, `mostcommon`,
       `leastcommon`, `initset`, `both`, `either`, `increment`,
       `decrement`, `crement`, `sign`, `positive`, `toivec`, `tojvec`,
       `sfilter`, `mfilter`, `extract`, `totuple`, `first`, `last`,
       `insert`, `remove`, `other`, `interval`, `astuple`, `product`,
       `pair`, `branch`, `compose`, `chain`, `matcher`, `rbind`,
       `lbind`, `power`, `fork`, `apply`, `rapply`, `mapply`, `papply`,
       `mpapply`, `prapply`.

### 4.2 Grid / Object / Geometry Primitives

`mostcolor`, `leastcolor`, `height`, `width`, `shape`,
       `portrait`, `colorcount`, `colorfilter`, `sizefilter`,
       `asindices`, `ofcolor`, `ulcorner`, `urcorner`, `llcorner`,
       `lrcorner`, `crop`, `toindices`, `recolor`, `shift`, `normalize`,
       `dneighbors`, `ineighbors`, `neighbors`, `objects`, `partition`,
       `fgpartition`, `uppermost`, `lowermost`, `leftmost`, `rightmost`,
       `square`, `vline`, `hline`, `hmatching`, `vmatching`,
       `manhattan`, `adjacent`, `bordering`, `centerofmass`, `palette`,
       `numcolors`, `color`, `toobject`, `asobject`, `rot90`, `rot180`,
       `rot270`, `hmirror`, `vmirror`, `dmirror`, `cmirror`, `fill`,
       `paint`, `underfill`, `underpaint`, `hupscale`, `vupscale`,
       `upscale`, `downscale`, `hconcat`, `vconcat`, `subgrid`,
       `hsplit`, `vsplit`, `cellwise`, `replace`, `switch`, `center`,
       `position`, `index`, `canvas`, `corners`, `connect`, `cover`,
       `trim`, `move`, `tophalf`, `bottomhalf`, `lefthalf`,
       `righthalf`, `vfrontier`, `hfrontier`, `backdrop`, `delta`,
       `gravitate`, `inbox`, `outbox`, `box`, `shoot`, `occurrences`,
       `frontiers`, `compress`, `hperiod`, `vperiod`.

In other words, the *current* DSL surface consists of:

* A rich set of **pure functional combinators and container ops**,
* A complete **grid / object geometry API** originally designed for ARC.

For the Gyan reasoning project, we will:

* Treat all 160 of these as **available primitives** (backbone).
* Add the **math-derived families** above (expressions, equations, etc.)
  in a way that composes with the existing functions rather than
  replacing them.

Because both this document and `dev/analyze_math_dsl.py` /
`dsl/dsl.py` are machine-checked, we have a single source of truth for
*all* DSL primitives (ARC-style + math-style) and can be confident we
are not silently missing any operation when we design training tasks.

## 5. Logic DSL

In addition to numeric / algebraic reasoning, we want Gyan to reason
over *logical structure itself*. The following primitives are proposed
for a general-purpose logic DSL. These are not derived from the
DeepMind math dataset; they are *design choices* to support
cross-domain reasoning experiments.

### 5.0 Logic Types

* `Prop`         – propositional formula
* `Bool`         – boolean value (`True` / `False`)
* `Term`         – first-order term (variable, constant, or function)
* `Constraint`   – generic constraint over variables / domains
* `State`        – CSP state (mapping from variables to domains/values)
* `Substitution` – mapping from variables to terms

### 5.1 Propositional Logic (10 primitives)

**Constructors:**

* `PROP_VAR(name: str) -> Prop`             – create propositional variable (p, q, r)
* `PROP_TRUE() -> Prop`                     – boolean constant ⊤
* `PROP_FALSE() -> Prop`                    – boolean constant ⊥

**Connectives:**

* `AND(a: Prop, b: Prop) -> Prop`           – conjunction a ∧ b
* `OR(a: Prop, b: Prop) -> Prop`            – disjunction a ∨ b
* `NOT(a: Prop) -> Prop`                    – negation ¬a
* `IMPLIES(a: Prop, b: Prop) -> Prop`       – implication a → b
* `IFF(a: Prop, b: Prop) -> Prop`           – biconditional a ↔ b

**Evaluation:**

* `EVAL_PROP(p: Prop, env: Dict[str, Bool]) -> Bool`
    – evaluate a propositional formula under a truth assignment
* `SIMPLIFY_PROP(p: Prop) -> Prop`
    – simplify a formula (e.g. eliminate double negation, propagate
      constants, normalize)

### 5.2 Inference Rules (8 primitives)

These represent *proof steps* over `Prop` formulas.

* `MODUS_PONENS(p: Prop, p_implies_q: Prop) -> Prop`
    – from p and (p → q), derive q
* `MODUS_TOLLENS(not_q: Prop, p_implies_q: Prop) -> Prop`
    – from ¬q and (p → q), derive ¬p
* `HYPOTHETICAL_SYLLOGISM(p_implies_q: Prop, q_implies_r: Prop) -> Prop`
    – from (p → q) and (q → r), derive (p → r)
* `DISJUNCTIVE_SYLLOGISM(p_or_q: Prop, not_p: Prop) -> Prop`
    – from (p ∨ q) and ¬p, derive q
* `CONJUNCTION_INTRO(p: Prop, q: Prop) -> Prop`
    – from p and q, derive (p ∧ q)
* `CONJUNCTION_ELIM_L(p_and_q: Prop) -> Prop`
    – from (p ∧ q), derive p
* `CONJUNCTION_ELIM_R(p_and_q: Prop) -> Prop`
    – from (p ∧ q), derive q
* `DOUBLE_NEG_ELIM(not_not_p: Prop) -> Prop`
    – from ¬¬p, derive p

### 5.3 Constraint Satisfaction (6 primitives)

These are for CSP-style reasoning (e.g. Sudoku-like tasks),
phrased in a domain-agnostic way.

* `DOMAIN(var: Var) -> Set[Value]`
    – get current domain of a variable
* `ASSIGN(var: Var, val: Value, state: State) -> State`
    – set variable to a value in the state
* `PROPAGATE(constraint: Constraint, state: State) -> State`
    – apply a constraint, pruning inconsistent values from domains
* `ELIMINATE(var: Var, val: Value, state: State) -> State`
    – remove a candidate value from a variable’s domain
* `IS_CONSISTENT(state: State) -> Bool`
    – check that no variable has an empty domain
* `BACKTRACK(state: State, checkpoint: State) -> State`
    – restore a previous state (for search/backtracking)

### 5.4 First-Order Logic (6 primitives)

These extend propositional logic with quantification and terms for
experiments that need unification and quantifiers.

* `FORALL(var: Var, domain: Set, prop: Prop) -> Prop`
    – universal quantification ∀x ∈ D. P(x)
* `EXISTS(var: Var, domain: Set, prop: Prop) -> Prop`
    – existential quantification ∃x ∈ D. P(x)
* `PREDICATE(name: str, args: Seq[Term]) -> Prop`
    – predicate application P(x, y, …)
* `SUBSTITUTE_TERM(prop: Prop, var: Var, term: Term) -> Prop`
    – capture-avoiding substitution of a term for a variable in a formula
* `UNIFY(t1: Term, t2: Term) -> Optional[Substitution]`
    – find a substitution making two terms equal (or return None)
* `APPLY_SUBST(subst: Substitution, prop: Prop) -> Prop`
    – apply a substitution to all free occurrences of variables in a formula

## 6. Summary

| Category | Count | Source |
|----------|-------|--------|
| **Math: Numeric Core** | 13 | DeepMind mathematics_dataset |
| **Math: Expression/Polynomial** | 9 | DeepMind mathematics_dataset |
| **Math: Equations/Inequalities** | 7 | DeepMind mathematics_dataset |
| **Math: Sets/Sequences/Ordering** | 3 | DeepMind mathematics_dataset |
| **Math: Probability/Combinatorics** | 5 | DeepMind mathematics_dataset |
| **ARC: Container/Logic** | ~60 | arc-dsl |
| **ARC: Grid/Geometry** | ~100 | arc-dsl |
| **Logic: Propositional** | 10 | Design choice |
| **Logic: Inference Rules** | 8 | Design choice |
| **Logic: CSP** | 6 | Design choice |
| **Logic: First-Order** | 6 | Design choice |
| **Total** | ~227 | |

These primitives are orthogonal but share types (`Bool`, `Var`, `Set`, `Seq`) so that
reasoning traces can naturally mix numeric, logical, and grid-based reasoning.


