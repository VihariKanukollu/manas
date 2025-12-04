from .arc_types import *


# Optional dependency for math DSL primitives. The ARC-style DSL above
# does not require sympy; the math DSL section at the bottom will import
# it lazily and raise a clear error if it is unavailable.
try:
    import sympy as _sp  # type: ignore[attr-defined]
    from sympy.core.relational import Relational as _Rel  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - environment without sympy
    _sp = None
    _Rel = None


def identity(
    x: Any
) -> Any:
    """ identity function """
    return x


def add(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ addition """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)


def subtract(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ subtraction """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)


def multiply(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ multiplication """
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)
    

def divide(
    a: Numerical,
    b: Numerical
) -> Numerical:
    """ floor division """
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)


def invert(
    n: Numerical
) -> Numerical:
    """ inversion with respect to addition """
    return -n if isinstance(n, int) else (-n[0], -n[1])


def even(
    n: Integer
) -> Boolean:
    """ evenness """
    return n % 2 == 0


def double(
    n: Numerical
) -> Numerical:
    """ scaling by two """
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)


def halve(
    n: Numerical
) -> Numerical:
    """ scaling by one half """
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)


def flip(
    b: Boolean
) -> Boolean:
    """ logical not """
    return not b


def equality(
    a: Any,
    b: Any
) -> Boolean:
    """ equality """
    return a == b


def contained(
    value: Any,
    container: Container
) -> Boolean:
    """ element of """
    return value in container


def combine(
    a: Container,
    b: Container
) -> Container:
    """ union """
    return type(a)((*a, *b))


def intersection(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ returns the intersection of two containers """
    return a & b


def difference(
    a: FrozenSet,
    b: FrozenSet
) -> FrozenSet:
    """ set difference """
    return type(a)(e for e in a if e not in b)


def dedupe(
    tup: Tuple
) -> Tuple:
    """ remove duplicates """
    return tuple(e for i, e in enumerate(tup) if tup.index(e) == i)


def order(
    container: Container,
    compfunc: Callable
) -> Tuple:
    """ order container by custom key """
    return tuple(sorted(container, key=compfunc))


def repeat(
    item: Any,
    num: Integer
) -> Tuple:
    """ repetition of item within vector """
    return tuple(item for i in range(num))


def greater(
    a: Integer,
    b: Integer
) -> Boolean:
    """ greater """
    return a > b


def size(
    container: Container
) -> Integer:
    """ cardinality """
    return len(container)


def merge(
    containers: ContainerContainer
) -> Container:
    """ merging """
    return type(containers)(e for c in containers for e in c)


def maximum(
    container: IntegerSet
) -> Integer:
    """ maximum """
    return max(container, default=0)


def minimum(
    container: IntegerSet
) -> Integer:
    """ minimum """
    return min(container, default=0)


def valmax(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ maximum by custom function """
    return compfunc(max(container, key=compfunc, default=0))


def valmin(
    container: Container,
    compfunc: Callable
) -> Integer:
    """ minimum by custom function """
    return compfunc(min(container, key=compfunc, default=0))


def argmax(
    container: Container,
    compfunc: Callable
) -> Any:
    """ largest item by custom order """
    return max(container, key=compfunc)


def argmin(
    container: Container,
    compfunc: Callable
) -> Any:
    """ smallest item by custom order """
    return min(container, key=compfunc)


def mostcommon(
    container: Container
) -> Any:
    """ most common item """
    return max(set(container), key=container.count)


def leastcommon(
    container: Container
) -> Any:
    """ least common item """
    return min(set(container), key=container.count)


def initset(
    value: Any
) -> FrozenSet:
    """ initialize container """
    return frozenset({value})


def both(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical and """
    return a and b


def either(
    a: Boolean,
    b: Boolean
) -> Boolean:
    """ logical or """
    return a or b


def increment(
    x: Numerical
) -> Numerical:
    """ incrementing """
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)


def decrement(
    x: Numerical
) -> Numerical:
    """ decrementing """
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)


def crement(
    x: Numerical
) -> Numerical:
    """ incrementing positive and decrementing negative """
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1)
    )


def sign(
    x: Numerical
) -> Numerical:
    """ sign """
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1)
    )


def positive(
    x: Integer
) -> Boolean:
    """ positive """
    return x > 0


def toivec(
    i: Integer
) -> IntegerTuple:
    """ vector pointing vertically """
    return (i, 0)


def tojvec(
    j: Integer
) -> IntegerTuple:
    """ vector pointing horizontally """
    return (0, j)


def sfilter(
    container: Container,
    condition: Callable
) -> Container:
    """ keep elements in container that satisfy condition """
    return type(container)(e for e in container if condition(e))


def mfilter(
    container: Container,
    function: Callable
) -> FrozenSet:
    """ filter and merge """
    return merge(sfilter(container, function))


def extract(
    container: Container,
    condition: Callable
) -> Any:
    """ first element of container that satisfies condition """
    return next(e for e in container if condition(e))


def totuple(
    container: FrozenSet
) -> Tuple:
    """ conversion to tuple """
    return tuple(container)


def first(
    container: Container
) -> Any:
    """ first item of container """
    return next(iter(container))


def last(
    container: Container
) -> Any:
    """ last item of container """
    return max(enumerate(container))[1]


def insert(
    value: Any,
    container: FrozenSet
) -> FrozenSet:
    """ insert item into container """
    return container.union(frozenset({value}))


def remove(
    value: Any,
    container: Container
) -> Container:
    """ remove item from container """
    return type(container)(e for e in container if e != value)


def other(
    container: Container,
    value: Any
) -> Any:
    """ other value in the container """
    return first(remove(value, container))


def interval(
    start: Integer,
    stop: Integer,
    step: Integer
) -> Tuple:
    """ range """
    return tuple(range(start, stop, step))


def astuple(
    a: Integer,
    b: Integer
) -> IntegerTuple:
    """ constructs a tuple """
    return (a, b)


def product(
    a: Container,
    b: Container
) -> FrozenSet:
    """ cartesian product """
    return frozenset((i, j) for j in b for i in a)


def pair(
    a: Tuple,
    b: Tuple
) -> TupleTuple:
    """ zipping of two tuples """
    return tuple(zip(a, b))


def branch(
    condition: Boolean,
    a: Any,
    b: Any
) -> Any:
    """ if else branching """
    return a if condition else b


def compose(
    outer: Callable,
    inner: Callable
) -> Callable:
    """ function composition """
    return lambda x: outer(inner(x))


def chain(
    h: Callable,
    g: Callable,
    f: Callable,
) -> Callable:
    """ function composition with three functions """
    return lambda x: h(g(f(x)))


def matcher(
    function: Callable,
    target: Any
) -> Callable:
    """ construction of equality function """
    return lambda x: function(x) == target


def rbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the rightmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)


def lbind(
    function: Callable,
    fixed: Any
) -> Callable:
    """ fix the leftmost argument """
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)


def power(
    function: Callable,
    n: Integer
) -> Callable:
    """ power of function """
    if n == 1:
        return function
    return compose(function, power(function, n - 1))


def fork(
    outer: Callable,
    a: Callable,
    b: Callable
) -> Callable:
    """ creates a wrapper function """
    return lambda x: outer(a(x), b(x))


def apply(
    function: Callable,
    container: Container
) -> Container:
    """ apply function to each item in container """
    return type(container)(function(e) for e in container)


def rapply(
    functions: Container,
    value: Any
) -> Container:
    """ apply each function in container to value """
    return type(functions)(function(value) for function in functions)


def mapply(
    function: Callable,
    container: ContainerContainer
) -> FrozenSet:
    """ apply and merge """
    return merge(apply(function, container))


def papply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors """
    return tuple(function(i, j) for i, j in zip(a, b))


def mpapply(
    function: Callable,
    a: Tuple,
    b: Tuple
) -> Tuple:
    """ apply function on two vectors and merge """
    # The second argument is conceptually a vector of containers, but in many
    # ARC-style usages it is passed as a (frozen)set of groups. Iteration order
    # over sets is not stable, which would make the semantics of mpapply
    # depend on hash order. To keep behaviour deterministic, we canonicalise
    # the order of `b` when it is not already a tuple.
    if isinstance(b, tuple):
        b_seq = b
    else:
        try:
            # Sort groups by a canonical representation of their contents.
            b_seq = tuple(sorted(b, key=lambda group: tuple(sorted(group))))
        except TypeError:
            # Fallback: best-effort conversion preserving whatever order the
            # container currently yields.
            b_seq = tuple(b)

    return merge(papply(function, a, b_seq))


def prapply(
    function,
    a: Container,
    b: Container
) -> FrozenSet:
    """ apply function on cartesian product """
    return frozenset(function(i, j) for j in b for i in a)


def mostcolor(
    element: Element
) -> Integer:
    """ most common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return max(set(values), key=values.count)
    

def leastcolor(
    element: Element
) -> Integer:
    """ least common color """
    values = [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    return min(set(values), key=values.count)


def height(
    piece: Piece
) -> Integer:
    """ height of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1


def width(
    piece: Piece
) -> Integer:
    """ width of grid or patch """
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1


def shape(
    piece: Piece
) -> IntegerTuple:
    """ height and width of grid or patch """
    return (height(piece), width(piece))


def portrait(
    piece: Piece
) -> Boolean:
    """ whether height is greater than width """
    return height(piece) > width(piece)


def colorcount(
    element: Element,
    value: Integer
) -> Integer:
    """ number of cells with color """
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)


def colorfilter(
    objs: Objects,
    value: Integer
) -> Objects:
    """ filter objects by color """
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)


def sizefilter(
    container: Container,
    n: Integer
) -> FrozenSet:
    """ filter items by size """
    return frozenset(item for item in container if len(item) == n)


def asindices(
    grid: Grid
) -> Indices:
    """ indices of all grid cells """
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))


def ofcolor(
    grid: Grid,
    value: Integer
) -> Indices:
    """ indices of all grid cells with value """
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)


def ulcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper left corner """
    return tuple(map(min, zip(*toindices(patch))))


def urcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of upper right corner """
    return tuple(map(lambda ix: {0: min, 1: max}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


def llcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower left corner """
    return tuple(map(lambda ix: {0: max, 1: min}[ix[0]](ix[1]), enumerate(zip(*toindices(patch)))))


def lrcorner(
    patch: Patch
) -> IntegerTuple:
    """ index of lower right corner """
    return tuple(map(max, zip(*toindices(patch))))


def crop(
    grid: Grid,
    start: IntegerTuple,
    dims: IntegerTuple
) -> Grid:
    """ subgrid specified by start and dimension """
    return tuple(r[start[1]:start[1]+dims[1]] for r in grid[start[0]:start[0]+dims[0]])


def toindices(
    patch: Patch
) -> Indices:
    """ indices of object cells """
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch


def recolor(
    value: Integer,
    patch: Patch
) -> Object:
    """ recolor patch """
    return frozenset((value, index) for index in toindices(patch))


def shift(
    patch: Patch,
    directions: IntegerTuple
) -> Patch:
    """ shift patch """
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)


def normalize(
    patch: Patch
) -> Patch:
    """ moves upper left corner to origin """
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))


def dneighbors(
    loc: IntegerTuple
) -> Indices:
    """ directly adjacent indices """
    return frozenset({(loc[0] - 1, loc[1]), (loc[0] + 1, loc[1]), (loc[0], loc[1] - 1), (loc[0], loc[1] + 1)})


def ineighbors(
    loc: IntegerTuple
) -> Indices:
    """ diagonally adjacent indices """
    return frozenset({(loc[0] - 1, loc[1] - 1), (loc[0] - 1, loc[1] + 1), (loc[0] + 1, loc[1] - 1), (loc[0] + 1, loc[1] + 1)})


def neighbors(
    loc: IntegerTuple
) -> Indices:
    """ adjacent indices """
    return dneighbors(loc) | ineighbors(loc)


def objects(
    grid: Grid,
    univalued: Boolean,
    diagonal: Boolean,
    without_bg: Boolean
) -> Objects:
    """ objects occurring on the grid """
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {
                        (i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w
                    }
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)


def partition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid)
    )


def fgpartition(
    grid: Grid
) -> Objects:
    """ each cell with the same value part of the same object without background """
    return frozenset(
        frozenset(
            (v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value
        ) for value in palette(grid) - {mostcolor(grid)}
    )


def uppermost(
    patch: Patch
) -> Integer:
    """ row index of uppermost occupied cell """
    return min(i for i, j in toindices(patch))


def lowermost(
    patch: Patch
) -> Integer:
    """ row index of lowermost occupied cell """
    return max(i for i, j in toindices(patch))


def leftmost(
    patch: Patch
) -> Integer:
    """ column index of leftmost occupied cell """
    return min(j for i, j in toindices(patch))


def rightmost(
    patch: Patch
) -> Integer:
    """ column index of rightmost occupied cell """
    return max(j for i, j in toindices(patch))


def square(
    piece: Piece
) -> Boolean:
    """ whether the piece forms a square """
    return len(piece) == len(piece[0]) if isinstance(piece, tuple) else height(piece) * width(piece) == len(piece) and height(piece) == width(piece)


def vline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a vertical line """
    return height(patch) == len(patch) and width(patch) == 1


def hline(
    patch: Patch
) -> Boolean:
    """ whether the piece forms a horizontal line """
    return width(patch) == len(patch) and height(patch) == 1


def hmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a row for which both patches have cells """
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0


def vmatching(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether there exists a column for which both patches have cells """
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0


def manhattan(
    a: Patch,
    b: Patch
) -> Integer:
    """ closest manhattan distance between two patches """
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))


def adjacent(
    a: Patch,
    b: Patch
) -> Boolean:
    """ whether two patches are adjacent """
    return manhattan(a, b) == 1


def bordering(
    patch: Patch,
    grid: Grid
) -> Boolean:
    """ whether a patch is adjacent to a grid border """
    return uppermost(patch) == 0 or leftmost(patch) == 0 or lowermost(patch) == len(grid) - 1 or rightmost(patch) == len(grid[0]) - 1


def centerofmass(
    patch: Patch
) -> IntegerTuple:
    """ center of mass """
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch))))


def palette(
    element: Element
) -> IntegerSet:
    """ colors occurring in object or grid """
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})


def numcolors(
    element: Element
) -> IntegerSet:
    """ number of colors occurring in object or grid """
    return len(palette(element))


def color(
    obj: Object
) -> Integer:
    """ color of object """
    return next(iter(obj))[0]


def toobject(
    patch: Patch,
    grid: Grid
) -> Object:
    """ object from patch and grid """
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)


def asobject(
    grid: Grid
) -> Object:
    """ conversion of grid to object """
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))


def rot90(
    grid: Grid
) -> Grid:
    """ quarter clockwise rotation """
    return tuple(row for row in zip(*grid[::-1]))


def rot180(
    grid: Grid
) -> Grid:
    """ half rotation """
    return tuple(tuple(row[::-1]) for row in grid[::-1])


def rot270(
    grid: Grid
) -> Grid:
    """ quarter anticlockwise rotation """
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1]))[::-1]


def hmirror(
    piece: Piece
) -> Piece:
    """ mirroring along horizontal """
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)


def vmirror(
    piece: Piece
) -> Piece:
    """ mirroring along vertical """
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)


def dmirror(
    piece: Piece
) -> Piece:
    """ mirroring along diagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*piece))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)


def cmirror(
    piece: Piece
) -> Piece:
    """ mirroring along counterdiagonal """
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1])))
    return vmirror(dmirror(vmirror(piece)))


def fill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices """
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)


def paint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid """
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)


def underfill(
    grid: Grid,
    value: Integer,
    patch: Patch
) -> Grid:
    """ fill value at indices that are background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def underpaint(
    grid: Grid,
    obj: Object
) -> Grid:
    """ paint object to grid where there is background """
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def hupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid horizontally """
    g = tuple()
    for row in grid:
        r = tuple()
        for value in row:
            r = r + tuple(value for num in range(factor))
        g = g + (r,)
    return g


def vupscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ upscale grid vertically """
    g = tuple()
    for row in grid:
        g = g + tuple(row for num in range(factor))
    return g


def upscale(
    element: Element,
    factor: Integer
) -> Element:
    """ upscale object or grid """
    if isinstance(element, tuple):
        g = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            g = g + tuple(upscaled_row for num in range(factor))
        return g
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        o = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    o.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(o), (di_inv, dj_inv))


def downscale(
    grid: Grid,
    factor: Integer
) -> Grid:
    """ downscale grid """
    h, w = len(grid), len(grid[0])
    g = tuple()
    for i in range(h):
        r = tuple()
        for j in range(w):
            if j % factor == 0:
                r = r + (grid[i][j],)
        g = g + (r, )
    h = len(g)
    dsg = tuple()
    for i in range(h):
        if i % factor == 0:
            dsg = dsg + (g[i],)
    return dsg


def hconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids horizontally """
    return tuple(i + j for i, j in zip(a, b))


def vconcat(
    a: Grid,
    b: Grid
) -> Grid:
    """ concatenate two grids vertically """
    return a + b


def subgrid(
    patch: Patch,
    grid: Grid
) -> Grid:
    """ smallest subgrid containing object """
    return crop(grid, ulcorner(patch), shape(patch))


def hsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid horizontally """
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))


def vsplit(
    grid: Grid,
    n: Integer
) -> Tuple:
    """ split grid vertically """
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))


def cellwise(
    a: Grid,
    b: Grid,
    fallback: Integer
) -> Grid:
    """ cellwise match of two grids """
    h, w = len(a), len(a[0])
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row, )
    return resulting_grid


def replace(
    grid: Grid,
    replacee: Integer,
    replacer: Integer
) -> Grid:
    """ color substitution """
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)


def switch(
    grid: Grid,
    a: Integer,
    b: Integer
) -> Grid:
    """ color switching """
    return tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid)


def center(
    patch: Patch
) -> IntegerTuple:
    """ center of the patch """
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)


def position(
    a: Patch,
    b: Patch
) -> IntegerTuple:
    """ relative position between two patches """
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)


def index(
    grid: Grid,
    loc: IntegerTuple
) -> Integer:
    """ color at location """
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]] 


def canvas(
    value: Integer,
    dimensions: IntegerTuple
) -> Grid:
    """ grid construction """
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))


def corners(
    patch: Patch
) -> Indices:
    """ indices of corners """
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})


def connect(
    a: IntegerTuple,
    b: IntegerTuple
) -> Indices:
    """ line between two points """
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej)))
    elif bi - ai == aj - bj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1)))
    return frozenset()


def cover(
    grid: Grid,
    patch: Patch
) -> Grid:
    """ remove object from grid """
    return fill(grid, mostcolor(grid), toindices(patch))


def trim(
    grid: Grid
) -> Grid:
    """ trim border of grid """
    return tuple(r[1:-1] for r in grid[1:-1])


def move(
    grid: Grid,
    obj: Object,
    offset: IntegerTuple
) -> Grid:
    """ move object on grid """
    return paint(cover(grid, obj), shift(obj, offset))


def tophalf(
    grid: Grid
) -> Grid:
    """ upper half of grid """
    return grid[:len(grid) // 2]


def bottomhalf(
    grid: Grid
) -> Grid:
    """ lower half of grid """
    return grid[len(grid) // 2 + len(grid) % 2:]


def lefthalf(
    grid: Grid
) -> Grid:
    """ left half of grid """
    return rot270(tophalf(rot90(grid)))


def righthalf(
    grid: Grid
) -> Grid:
    """ right half of grid """
    return rot270(bottomhalf(rot90(grid)))


def vfrontier(
    location: IntegerTuple
) -> Indices:
    """ vertical frontier """
    return frozenset((i, location[1]) for i in range(30))


def hfrontier(
    location: IntegerTuple
) -> Indices:
    """ horizontal frontier """
    return frozenset((location[0], j) for j in range(30))


def backdrop(
    patch: Patch
) -> Indices:
    """ indices in bounding box of patch """
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))


def delta(
    patch: Patch
) -> Indices:
    """ indices in bounding box but not part of patch """
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)


def gravitate(
    source: Patch,
    destination: Patch
) -> IntegerTuple:
    """ direction to move source until adjacent to destination """
    si, sj = center(source)
    di, dj = center(destination)
    i, j = 0, 0
    if vmatching(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacent(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shift(source, (i, j))
    return (gi - i, gj - j)


def inbox(
    patch: Patch
) -> Indices:
    """ inbox for patch """
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def outbox(
    patch: Patch
) -> Indices:
    """ outbox for patch """
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def box(
    patch: Patch
) -> Indices:
    """ outline of patch """
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def shoot(
    start: IntegerTuple,
    direction: IntegerTuple
) -> Indices:
    """ line from starting point and direction """
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))


def occurrences(
    grid: Grid,
    obj: Object
) -> Indices:
    """ locations of occurrences of object in grid """
    occs = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    oh, ow = shape(obj)
    h2, w2 = h - oh + 1, w - ow + 1
    for i in range(h2):
        for j in range(w2):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if not (0 <= a < h and 0 <= b < w and grid[a][b] == v):
                    occurs = False
                    break
            if occurs:
                occs.add((i, j))
    return frozenset(occs)


def frontiers(
    grid: Grid
) -> Objects:
    """ set of frontiers """
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices})
    vfrontiers = frozenset({frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices})
    return hfrontiers | vfrontiers


def compress(
    grid: Grid
) -> Grid:
    """ removes frontiers from grid """
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri)


def hperiod(
    obj: Object
) -> Integer:
    """ horizontal periodicity """
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w


def vperiod(
    obj: Object
) -> Integer:
    """ vertical periodicity """
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h


# ============================================================================
# Formal math DSL (Gyan)
# ============================================================================

# We keep the implementation here intentionally thin: these primitives are
# mostly light wrappers around sympy, matching the abstract spec in
# dsl_spec.md. They are used by data generators / checkers and to define
# what "one reasoning step" means; the model itself will learn to emit
# *textual* DSL programs using these names.


# --- Type aliases -----------------------------------------------------------

if _sp is not None:
    Expr = _sp.Expr          # generic sympy expression
    Poly = _sp.Poly          # polynomial in one variable
    Var = _sp.Symbol         # variable symbol
    Equation = _sp.Equality  # sympy.Eq instance
    Ineq = _Rel              # generic inequality
else:
    # Fallback types when sympy is not available; functions will raise at use.
    Expr = Any
    Poly = Any
    Var = Any
    Equation = Any
    Ineq = Any


def _require_sympy() -> None:
    if _sp is None:
        raise RuntimeError(
            "sympy is required for the formal math DSL, but it is not installed."
        )


# --- Numeric core -----------------------------------------------------------

def gcd(
    a: Integer,
    b: Integer
) -> Integer:
    """Greatest common divisor."""
    _require_sympy()
    return int(_sp.gcd(int(a), int(b)))


def lcm(
    a: Integer,
    b: Integer
) -> Integer:
    """Least common multiple."""
    _require_sympy()
    return int(_sp.lcm(int(a), int(b)))


def factorint(
    n: Integer
) -> dict:
    """Prime factorization as {prime: exponent}."""
    _require_sympy()
    # sympy.factorint returns a dict-like mapping already.
    return dict(_sp.factorint(int(n)))


def factorial(
    n: Integer
) -> Integer:
    """Factorial n!."""
    _require_sympy()
    return int(_sp.factorial(int(n)))


def prod(
    seq: Container[Num]
) -> Num:
    """Product of a finite sequence."""
    _require_sympy()
    return _sp.prod(seq)


def is_integer(
    x: Any
) -> Boolean:
    """Predicate: is x an integer-valued quantity?"""
    if isinstance(x, int):
        return True
    if _sp is not None and isinstance(x, _sp.Basic):
        return bool(x.is_integer)
    return False


# --- Expressions / polynomials ----------------------------------------------

def var(
    name: str
) -> Var:
    """Create a symbolic variable."""
    _require_sympy()
    return _sp.Symbol(name)


def const(
    value: Num
) -> Expr:
    """Create a constant expression from a numeric value."""
    _require_sympy()
    if isinstance(value, int):
        return _sp.Integer(value)
    return _sp.Rational(value)


def make_poly(
    coeffs: Tuple[Num, ...],
    variable: Var
) -> Poly:
    """
    Build a univariate polynomial from coefficients in increasing degree order.

    For example:
        coeffs = (1, 2, 3)  ->  1 + 2*x + 3*x**2
    """
    _require_sympy()
    expr = sum(const(c) * variable ** i for i, c in enumerate(coeffs))
    return _sp.Poly(expr, variable)


def eval_expr(
    expr: Expr,
    env: dict
) -> Expr:
    """Evaluate an expression under a substitution environment."""
    _require_sympy()
    return expr.subs(env)


def expand_expr(
    expr: Expr
) -> Expr:
    """Algebraic expansion."""
    _require_sympy()
    return _sp.expand(expr)


def factor_expr(
    expr: Expr
) -> Expr:
    """Algebraic factorization."""
    _require_sympy()
    return _sp.factor(expr)


def simplify_expr(
    expr: Expr
) -> Expr:
    """Simplification of an expression."""
    _require_sympy()
    return _sp.simplify(expr)


def differentiate(
    expr: Expr,
    variable: Var,
    order: Integer = 1
) -> Expr:
    """Differentiate expr w.r.t. variable, possibly higher order."""
    _require_sympy()
    return _sp.diff(expr, variable, int(order))


def integrate_poly(
    expr: Expr,
    variable: Var
) -> Expr:
    """Integrate a (polynomial) expression w.r.t. variable."""
    _require_sympy()
    return _sp.integrate(expr, variable)


# --- Equations and inequalities ---------------------------------------------

def eq(
    lhs: Expr,
    rhs: Expr
) -> Equation:
    """Construct an equation lhs == rhs."""
    _require_sympy()
    return _sp.Eq(lhs, rhs)


def ineq(
    op: str,
    lhs: Expr,
    rhs: Expr
) -> Ineq:
    """Construct an inequality lhs (op) rhs, where op in {'<','<=','>','>=','==','!='}."""
    _require_sympy()
    table = {
        "<": _sp.Lt,
        "<=": _sp.Le,
        ">": _sp.Gt,
        ">=": _sp.Ge,
        "==": _sp.Eq,
        "!=": _sp.Ne,
    }
    assert op in table, f"Unsupported inequality op: {op}"
    return table[op](lhs, rhs)


def _lift_rel(
    rel: Ineq,
    new_lhs: Expr,
    new_rhs: Expr
) -> Ineq:
    """Rebuild a relational object of the same type with new sides."""
    _require_sympy()
    if isinstance(rel, _sp.Equality):
        return _sp.Eq(new_lhs, new_rhs)
    # For generic relational, use its class constructor.
    return rel.func(new_lhs, new_rhs)


def add_both_sides(
    rel: Ineq,
    delta: Expr
) -> Ineq:
    """Add delta to both sides of an equation/inequality."""
    _require_sympy()
    return _lift_rel(rel, rel.lhs + delta, rel.rhs + delta)


def mul_both_sides(
    rel: Ineq,
    factor: Expr
) -> Ineq:
    """Multiply both sides of an equation/inequality by factor."""
    _require_sympy()
    return _lift_rel(rel, rel.lhs * factor, rel.rhs * factor)


def div_both_sides(
    rel: Ineq,
    factor: Expr
) -> Ineq:
    """Divide both sides of an equation/inequality by factor."""
    _require_sympy()
    return _lift_rel(rel, rel.lhs / factor, rel.rhs / factor)


def substitute(
    rel: Ineq,
    variable: Var,
    repl: Expr
) -> Ineq:
    """Substitute variable -> repl on both sides of an equation/inequality."""
    _require_sympy()
    return _lift_rel(rel, rel.lhs.subs(variable, repl), rel.rhs.subs(variable, repl))


def is_solution(
    rel: Ineq,
    variable: Var,
    value: Num
) -> Boolean:
    """
    Check whether value is a solution to an equation / inequality.

    For equations this means lhs == rhs after substitution; for
    inequalities, the relational must evaluate to True.
    """
    _require_sympy()
    substituted = rel.subs(variable, value)
    # SymPy may return BooleanTrue/BooleanFalse or an expression; bool()
    # forces evaluation when possible.
    try:
        return bool(substituted)
    except TypeError:
        # If SymPy cannot decide, treat as not a solution.
        return False


# --- Ordering / selection ---------------------------------------------------

def sort(
    seq: Tuple[Num, ...]
) -> Tuple[Num, ...]:
    """Sort a finite numeric sequence in ascending order."""
    return tuple(sorted(seq))


def kth_largest(
    seq: Tuple[Num, ...],
    k: Integer
) -> Num:
    """Return the k-th largest element (1-indexed)."""
    assert len(seq) >= int(k) > 0, "k must be between 1 and len(seq)"
    ordered = sorted(seq, reverse=True)
    return ordered[int(k) - 1]


def closest_to(
    seq: Tuple[Num, ...],
    target: Num
) -> Num:
    """Return element of seq with minimal absolute distance to target."""
    return min(seq, key=lambda x: abs(x - target))


# ============================================================================
# Logic DSL (Gyan)
# ============================================================================

# Propositional formulas are represented as nested tuples:
#   ("VAR", name)
#   ("TRUE",)
#   ("FALSE",)
#   ("AND", a, b)
#   ("OR", a, b)
#   ("NOT", a)
#   ("IMPLIES", a, b)
#   ("IFF", a, b)
#
# This is a lightweight representation that doesn't require sympy.logic.

Prop = Tuple  # Propositional formula (nested tuple)


# --- 5.1 Propositional Logic (10 primitives) --------------------------------

def PROP_VAR(name: str) -> Prop:
    """Create a propositional variable."""
    return ("VAR", name)


def PROP_TRUE() -> Prop:
    """Boolean constant True."""
    return ("TRUE",)


def PROP_FALSE() -> Prop:
    """Boolean constant False."""
    return ("FALSE",)


def AND(a: Prop, b: Prop) -> Prop:
    """Conjunction a ∧ b."""
    return ("AND", a, b)


def OR(a: Prop, b: Prop) -> Prop:
    """Disjunction a ∨ b."""
    return ("OR", a, b)


def NOT(a: Prop) -> Prop:
    """Negation ¬a."""
    return ("NOT", a)


def IMPLIES(a: Prop, b: Prop) -> Prop:
    """Implication a → b."""
    return ("IMPLIES", a, b)


def IFF(a: Prop, b: Prop) -> Prop:
    """Biconditional a ↔ b."""
    return ("IFF", a, b)


def EVAL_PROP(p: Prop, env: dict) -> bool:
    """Evaluate a propositional formula under a truth assignment."""
    tag = p[0]
    if tag == "VAR":
        return env[p[1]]
    elif tag == "TRUE":
        return True
    elif tag == "FALSE":
        return False
    elif tag == "AND":
        return EVAL_PROP(p[1], env) and EVAL_PROP(p[2], env)
    elif tag == "OR":
        return EVAL_PROP(p[1], env) or EVAL_PROP(p[2], env)
    elif tag == "NOT":
        return not EVAL_PROP(p[1], env)
    elif tag == "IMPLIES":
        return (not EVAL_PROP(p[1], env)) or EVAL_PROP(p[2], env)
    elif tag == "IFF":
        return EVAL_PROP(p[1], env) == EVAL_PROP(p[2], env)
    raise ValueError(f"Unknown prop tag: {tag}")


def SIMPLIFY_PROP(p: Prop) -> Prop:
    """Simplify a propositional formula (basic rules)."""
    tag = p[0]
    if tag in ("VAR", "TRUE", "FALSE"):
        return p
    elif tag == "NOT":
        inner = SIMPLIFY_PROP(p[1])
        if inner[0] == "TRUE":
            return PROP_FALSE()
        if inner[0] == "FALSE":
            return PROP_TRUE()
        if inner[0] == "NOT":
            return inner[1]  # double negation elimination
        return ("NOT", inner)
    elif tag == "AND":
        a, b = SIMPLIFY_PROP(p[1]), SIMPLIFY_PROP(p[2])
        if a[0] == "FALSE" or b[0] == "FALSE":
            return PROP_FALSE()
        if a[0] == "TRUE":
            return b
        if b[0] == "TRUE":
            return a
        return ("AND", a, b)
    elif tag == "OR":
        a, b = SIMPLIFY_PROP(p[1]), SIMPLIFY_PROP(p[2])
        if a[0] == "TRUE" or b[0] == "TRUE":
            return PROP_TRUE()
        if a[0] == "FALSE":
            return b
        if b[0] == "FALSE":
            return a
        return ("OR", a, b)
    elif tag == "IMPLIES":
        a, b = SIMPLIFY_PROP(p[1]), SIMPLIFY_PROP(p[2])
        if a[0] == "FALSE" or b[0] == "TRUE":
            return PROP_TRUE()
        if a[0] == "TRUE":
            return b
        return ("IMPLIES", a, b)
    elif tag == "IFF":
        a, b = SIMPLIFY_PROP(p[1]), SIMPLIFY_PROP(p[2])
        if a == b:
            return PROP_TRUE()
        return ("IFF", a, b)
    return p


# --- 5.2 Inference Rules (8 primitives) -------------------------------------

def MODUS_PONENS(p: Prop, p_implies_q: Prop) -> Prop:
    """From p and (p → q), derive q."""
    assert p_implies_q[0] == "IMPLIES", "Second arg must be implication"
    assert p == p_implies_q[1], "First arg must match antecedent"
    return p_implies_q[2]


def MODUS_TOLLENS(not_q: Prop, p_implies_q: Prop) -> Prop:
    """From ¬q and (p → q), derive ¬p."""
    assert not_q[0] == "NOT", "First arg must be negation"
    assert p_implies_q[0] == "IMPLIES", "Second arg must be implication"
    assert not_q[1] == p_implies_q[2], "Negated must match consequent"
    return NOT(p_implies_q[1])


def HYPOTHETICAL_SYLLOGISM(p_implies_q: Prop, q_implies_r: Prop) -> Prop:
    """From (p → q) and (q → r), derive (p → r)."""
    assert p_implies_q[0] == "IMPLIES" and q_implies_r[0] == "IMPLIES"
    assert p_implies_q[2] == q_implies_r[1], "Consequent must match antecedent"
    return IMPLIES(p_implies_q[1], q_implies_r[2])


def DISJUNCTIVE_SYLLOGISM(p_or_q: Prop, not_p: Prop) -> Prop:
    """From (p ∨ q) and ¬p, derive q."""
    assert p_or_q[0] == "OR", "First arg must be disjunction"
    assert not_p[0] == "NOT", "Second arg must be negation"
    assert not_p[1] == p_or_q[1], "Negated must match first disjunct"
    return p_or_q[2]


def CONJUNCTION_INTRO(p: Prop, q: Prop) -> Prop:
    """From p and q, derive (p ∧ q)."""
    return AND(p, q)


def CONJUNCTION_ELIM_L(p_and_q: Prop) -> Prop:
    """From (p ∧ q), derive p."""
    assert p_and_q[0] == "AND", "Arg must be conjunction"
    return p_and_q[1]


def CONJUNCTION_ELIM_R(p_and_q: Prop) -> Prop:
    """From (p ∧ q), derive q."""
    assert p_and_q[0] == "AND", "Arg must be conjunction"
    return p_and_q[2]


def DOUBLE_NEG_ELIM(not_not_p: Prop) -> Prop:
    """From ¬¬p, derive p."""
    assert not_not_p[0] == "NOT" and not_not_p[1][0] == "NOT"
    return not_not_p[1][1]


# --- 5.3 Constraint Satisfaction (6 primitives) -----------------------------

# CSP State is a dict: { var_name: set_of_possible_values }
# Constraint is a callable: (state) -> state (with reduced domains)

CSPState = dict  # {str: set}
Constraint = Callable  # (CSPState) -> CSPState


def DOMAIN(var: str, state: CSPState) -> FrozenSet:
    """Get current domain of a variable."""
    return frozenset(state.get(var, set()))


def ASSIGN(var: str, val: Any, state: CSPState) -> CSPState:
    """Assign a single value to a variable."""
    new_state = dict(state)
    new_state[var] = {val}
    return new_state


def PROPAGATE(constraint: Constraint, state: CSPState) -> CSPState:
    """Apply a constraint, pruning inconsistent values from domains."""
    return constraint(state)


def ELIMINATE(var: str, val: Any, state: CSPState) -> CSPState:
    """Remove a candidate value from a variable's domain."""
    new_state = dict(state)
    new_state[var] = state[var] - {val}
    return new_state


def IS_CONSISTENT(state: CSPState) -> bool:
    """Check that no variable has an empty domain."""
    return all(len(d) > 0 for d in state.values())


def BACKTRACK(state: CSPState, checkpoint: CSPState) -> CSPState:
    """Restore a previous state (for search/backtracking)."""
    return dict(checkpoint)


# --- 5.3b Sudoku / Grid CSP primitives --------------------------------------

# Sudoku grid is represented as a tuple of tuples (9x9) or flat tuple (81 elements)
# Values 1-9 are filled cells, 0 represents empty cells

SudokuGrid = Tuple  # 9x9 grid as tuple of tuples, or flat 81-tuple


def _parse_sudoku_grid(grid: Any) -> Tuple[Tuple[int, ...], ...]:
    """
    Normalize a Sudoku grid to a 9x9 tuple of tuples.
    
    Accepts:
    - 9x9 tuple of tuples
    - 81-element flat tuple/list
    - String of 81 digits
    """
    if isinstance(grid, str):
        if len(grid) == 81:
            flat = [int(c) for c in grid]
            return tuple(tuple(flat[i*9:(i+1)*9]) for i in range(9))
        raise ValueError(f"String grid must be 81 chars, got {len(grid)}")
    
    if isinstance(grid, (tuple, list)):
        if len(grid) == 81:
            # Flat representation
            return tuple(tuple(grid[i*9:(i+1)*9]) for i in range(9))
        elif len(grid) == 9 and all(len(row) == 9 for row in grid):
            # Already 9x9
            return tuple(tuple(row) for row in grid)
    
    raise ValueError(f"Invalid grid format: expected 9x9 or 81-element sequence")


def CHECK_SUDOKU(grid: SudokuGrid) -> bool:
    """
    Check if a 9x9 Sudoku grid is valid.
    
    A valid Sudoku has:
    - Each row contains 1-9 exactly once
    - Each column contains 1-9 exactly once
    - Each 3x3 box contains 1-9 exactly once
    
    Note: This checks COMPLETED grids (no zeros/blanks).
    """
    try:
        g = _parse_sudoku_grid(grid)
    except ValueError:
        return False
    
    expected = set(range(1, 10))
    
    # Check rows
    for row in g:
        if set(row) != expected:
            return False
    
    # Check columns
    for col in range(9):
        col_vals = set(g[row][col] for row in range(9))
        if col_vals != expected:
            return False
    
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box_vals = set()
            for i in range(3):
                for j in range(3):
                    box_vals.add(g[box_row * 3 + i][box_col * 3 + j])
            if box_vals != expected:
                return False
    
    return True


def CHECK_LATIN(grid: Tuple) -> bool:
    """
    Check if a grid is a valid Latin square.
    
    Each row and column contains each value exactly once.
    (No 3x3 box constraint like Sudoku.)
    """
    try:
        g = _parse_sudoku_grid(grid)
    except ValueError:
        return False
    
    n = len(g)
    expected = set(range(1, n + 1))
    
    # Check rows
    for row in g:
        if set(row) != expected:
            return False
    
    # Check columns
    for col in range(n):
        col_vals = set(g[row][col] for row in range(n))
        if col_vals != expected:
            return False
    
    return True


def ALL_DIFFERENT(values: Container) -> bool:
    """Check that all values in a container are distinct."""
    vals = list(values)
    return len(vals) == len(set(vals))


def ROW_VALID(grid: SudokuGrid, row_idx: int) -> bool:
    """Check if a specific row in a Sudoku grid is valid (1-9, no repeats)."""
    try:
        g = _parse_sudoku_grid(grid)
    except ValueError:
        return False
    
    if not (0 <= row_idx < 9):
        return False
    
    row = g[row_idx]
    # Filter out zeros (empty cells)
    filled = [v for v in row if v != 0]
    return len(filled) == len(set(filled))


def COL_VALID(grid: SudokuGrid, col_idx: int) -> bool:
    """Check if a specific column in a Sudoku grid is valid (1-9, no repeats)."""
    try:
        g = _parse_sudoku_grid(grid)
    except ValueError:
        return False
    
    if not (0 <= col_idx < 9):
        return False
    
    col = [g[row][col_idx] for row in range(9)]
    filled = [v for v in col if v != 0]
    return len(filled) == len(set(filled))


def BOX_VALID(grid: SudokuGrid, box_idx: int) -> bool:
    """
    Check if a specific 3x3 box in a Sudoku grid is valid.
    
    Box indices (0-8):
        0 1 2
        3 4 5
        6 7 8
    """
    try:
        g = _parse_sudoku_grid(grid)
    except ValueError:
        return False
    
    if not (0 <= box_idx < 9):
        return False
    
    box_row = box_idx // 3
    box_col = box_idx % 3
    
    box_vals = []
    for i in range(3):
        for j in range(3):
            box_vals.append(g[box_row * 3 + i][box_col * 3 + j])
    
    filled = [v for v in box_vals if v != 0]
    return len(filled) == len(set(filled))


def CELL(grid: SudokuGrid, row: int, col: int) -> int:
    """Get the value at a specific cell in a Sudoku grid."""
    g = _parse_sudoku_grid(grid)
    return g[row][col]


def SET_CELL(grid: SudokuGrid, row: int, col: int, val: int) -> Tuple:
    """Set a cell value in a Sudoku grid, returning a new grid."""
    g = _parse_sudoku_grid(grid)
    new_grid = [list(r) for r in g]
    new_grid[row][col] = val
    return tuple(tuple(r) for r in new_grid)


def EMPTY_CELLS(grid: SudokuGrid) -> Tuple:
    """Return a tuple of (row, col) positions of empty cells (value 0)."""
    g = _parse_sudoku_grid(grid)
    empty = []
    for row in range(9):
        for col in range(9):
            if g[row][col] == 0:
                empty.append((row, col))
    return tuple(empty)


def CANDIDATES(grid: SudokuGrid, row: int, col: int) -> FrozenSet:
    """
    Return the set of valid candidate values for a cell.
    
    Returns values 1-9 that don't conflict with the row, column, or 3x3 box.
    """
    g = _parse_sudoku_grid(grid)
    
    if g[row][col] != 0:
        # Cell already filled
        return frozenset()
    
    used = set()
    
    # Row
    for c in range(9):
        used.add(g[row][c])
    
    # Column
    for r in range(9):
        used.add(g[r][col])
    
    # 3x3 box
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            used.add(g[box_row + i][box_col + j])
    
    return frozenset(range(1, 10)) - used


# --- 5.4 First-Order Logic (6 primitives) -----------------------------------

# FOL terms and formulas extend propositional logic.
# Term: ("CONST", value) | ("FVAR", name) | ("FUNC", name, args...)
# Prop extended: ("PRED", name, args...) | ("FORALL", var, domain, prop) | ("EXISTS", var, domain, prop)

Term = Tuple
Substitution = dict  # {var_name: Term}


def FORALL(var: str, domain: FrozenSet, prop: Prop) -> Prop:
    """Universal quantification ∀x ∈ D. P(x)."""
    return ("FORALL", var, domain, prop)


def EXISTS(var: str, domain: FrozenSet, prop: Prop) -> Prop:
    """Existential quantification ∃x ∈ D. P(x)."""
    return ("EXISTS", var, domain, prop)


def PREDICATE(name: str, args: Tuple) -> Prop:
    """Predicate application P(x, y, ...)."""
    return ("PRED", name) + args


def SUBSTITUTE_TERM(prop: Prop, var: str, term: Term) -> Prop:
    """Substitute a term for a variable in a formula."""
    tag = prop[0]
    if tag == "VAR":
        return term if prop[1] == var else prop
    elif tag in ("TRUE", "FALSE"):
        return prop
    elif tag == "NOT":
        return ("NOT", SUBSTITUTE_TERM(prop[1], var, term))
    elif tag in ("AND", "OR", "IMPLIES", "IFF"):
        return (tag, SUBSTITUTE_TERM(prop[1], var, term), SUBSTITUTE_TERM(prop[2], var, term))
    elif tag == "PRED":
        new_args = tuple(term if (a[0] == "FVAR" and a[1] == var) else a for a in prop[2:])
        return ("PRED", prop[1]) + new_args
    elif tag in ("FORALL", "EXISTS"):
        if prop[1] == var:
            return prop  # shadowed
        return (tag, prop[1], prop[2], SUBSTITUTE_TERM(prop[3], var, term))
    return prop


def UNIFY(t1: Term, t2: Term) -> Substitution:
    """Find a substitution making two terms equal, or None if impossible."""
    if t1 == t2:
        return {}
    if t1[0] == "FVAR":
        return {t1[1]: t2}
    if t2[0] == "FVAR":
        return {t2[1]: t1}
    if t1[0] == "FUNC" and t2[0] == "FUNC" and t1[1] == t2[1] and len(t1) == len(t2):
        subst = {}
        for a1, a2 in zip(t1[2:], t2[2:]):
            s = UNIFY(a1, a2)
            if s is None:
                return None
            subst.update(s)
        return subst
    return None


def APPLY_SUBST(subst: Substitution, prop: Prop) -> Prop:
    """Apply a substitution to all free occurrences of variables in a formula."""
    result = prop
    for var, term in subst.items():
        result = SUBSTITUTE_TERM(result, var, term)
    return result
