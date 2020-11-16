import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from heapq import heappop, heappush
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pulp as pl
from cytoolz.curried import take
from cytoolz.dicttoolz import assoc
from cytoolz.functoolz import compose
from cytoolz.itertoolz import first, second, topk, unique


def lpt(R: int, p: Sequence[float]) -> Tuple[List[int], float, float]:
    """
    LPT (Longest Processing Time First) approximation algorithm.

    >>> lpt(2, [])
    ([], 0.0, 1.0)
    >>> lpt(0, [10, 20])
    ([], 0.0, nan)
    >>> lpt(1, [10, 20])
    ([0, 0], 30.0, 2.0)

    >>> lpt(R=3, p=[5, 5, 4, 4, 3, 3, 3])
    ([0, 1, 2, 2, 0, 1, 0], 11.0, 1.4444444444444444)
    """
    if R < 1:
        return [], 0.0, float('nan')
    pts = sorted(enumerate(p), key=second, reverse=True)
    schedule, _, makespan, r = _lpt(R, pts)
    return schedule, makespan, r


def _lpt(
    R: int, p: Sequence[Tuple[int, float]]
) -> Tuple[List[int], np.ndarray, float, float]:
    cts = np.zeros(R, np.float64)
    hist = [0] * R

    schedule = [0] * len(p)
    for t, pt in p:
        i: int = (cts + pt).argmin()
        cts[i] += pt
        hist[i] += 1
        schedule[t] = i

    k = max(hist)
    r = 1 + (1 / k) + (1 / (k * R)) if k > 0 else 1.0
    return schedule, cts, np.max(cts), r


@dataclass(frozen=True)
class Solution:
    schedule: Dict[int, int]
    cts: np.ndarray  # resource completion times
    rts: np.ndarray  # remaining task times
    value: float
    h_value: float
    depth: int

    def __lt__(self, other: 'Solution') -> bool:
        return self.h_value < other.h_value

    def iter_pruned_resources(self) -> Iterable[int]:
        return map(first, unique(enumerate(self.cts), key=second))

    @property
    def task_allocation(self) -> List[int]:
        alloc = [0] * len(self.schedule)
        for j, i in self.schedule.items():
            alloc[j] = i
        return alloc

    @classmethod
    def lpt(cls, R: int, p: List[Tuple[int, float]]) -> 'Solution':
        n = len(p)
        schedule, cts, value, _ = _lpt(R, p)
        return cls(
            schedule=dict(enumerate(schedule)),
            cts=cts,
            rts=np.zeros(n, np.float64),
            value=value,
            h_value=value,
            depth=n,
        )

    @classmethod
    def init(cls, R: int, p: Sequence[float], h_value: float) -> 'Solution':
        return cls(
            schedule={},
            cts=np.zeros(R, np.float64),
            rts=np.array(p),
            value=0.0,
            h_value=h_value,
            depth=0,
        )


def h1(R: int, rts: np.ndarray, cts: np.ndarray) -> float:
    cts = cts.copy()
    time_left = np.sum(rts)

    while time_left > 0:

        if np.all(cts == cts[0]):
            # distribute tasks uniformly if all completion times are the same
            return cast(float, cts[0] + (time_left / R))

        a_min = np.argmin(cts)
        a_max = np.argmax(cts)
        diff = cts[a_max] - cts[a_min]

        cts[a_min] += diff
        time_left -= diff

    return cast(float, np.max(cts))


def h2(R: int, rts: np.ndarray, cts: np.ndarray) -> float:
    return cast(float, max(np.max(rts), (np.sum(cts) + np.sum(rts)) / R))


Heuristic = Callable[[int, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Stats:
    R: int
    n: int
    makespan: float
    max_open: Optional[int]
    expanded: int
    pruned: int
    pruned_value: int
    pruned_h_value: int
    pruned_closed: int
    proved_optimal: bool
    elapsed: timedelta

    @property
    def space_size(self) -> int:
        return cast(int, self.R ** self.n)

    @property
    def exhaustiveness_ratio(self) -> float:
        return self.expanded / self.space_size if self.space_size > 0 else 1

    @property
    def pruned_total(self) -> int:
        return self.pruned_value + self.pruned_h_value + self.pruned_closed


def bnb(
    R: int, p: Sequence[float], h: Heuristic, limit: Optional[timedelta] = None
) -> Tuple[List[int], Stats]:
    """
    Searches for optimal solution for `P || C_max` using Best-first BnB.

    >>> bnb(R=2, p=[], h=h1)  # doctest: +ELLIPSIS
    ([], Stats(..., makespan=0.0, ...))
    >>> bnb(R=0, p=[10, 20], h=h1)  # doctest: +ELLIPSIS
    ([], Stats(..., makespan=0.0, ...))

    >>> R = 3
    >>> schedule, stats = bnb(R, p=[5, 5, 4, 4, 3, 3, 3], h=h1)
    >>> stats.makespan
    9.0
    >>> stats.proved_optimal
    True

    There might be multiple symmetric optima, so we check just the
    task distribution.

    >>> sorted(sum(1 for r in schedule if r == i) for i in range(R))
    [2, 2, 3]
    """
    start = datetime.now()

    n = len(p)

    max_open = None
    expanded = 0
    pruned = 0
    pruned_value = 0
    pruned_h_value = 0
    pruned_closed = 0
    proved_optimal = True

    # pre-sorted set of tasks:
    # 1. LPT does it anyway
    # 2. Heuristic: larger tasks might prude the space sooner
    pts = sorted(enumerate(p), key=second, reverse=True)

    if R < 1:
        stats = Stats(
            R=R,
            n=n,
            makespan=0.0,
            max_open=max_open,
            expanded=expanded,
            pruned=pruned,
            pruned_value=pruned_value,
            pruned_h_value=0,
            pruned_closed=pruned_closed,
            proved_optimal=proved_optimal,
            elapsed=datetime.now() - start,
        )
        return [], stats

    closed = set()

    best = Solution.lpt(R, pts)

    queue: List[Solution] = []
    heappush(queue, Solution.init(R, p, best.value))

    # Best-first search using f(N) = h(N)
    while queue:
        max_open = max(max_open or 0, len(queue))

        # return current best if running for more than given time limit
        if limit is not None and (datetime.now() - start) > limit:
            proved_optimal = False
            break

        node: Solution = heappop(queue)
        expanded += 1

        if len(node.schedule) == len(pts):  # complete solution
            if node.value < best.value:  # found new minimum
                best = node
        else:  # inner node -> extend partial solution
            depth = node.depth + 1

            # heuristic: pick the biggest job that restricts the space the most
            j, pt = next((j, pt) for j, pt in pts if j not in node.schedule)

            # TODO: check - compare with node.iter_pruned_resources()
            _, rs = np.unique(node.cts, return_index=True)
            assert frozenset(rs) == frozenset(
                node.iter_pruned_resources()
            ), f'{rs} != {list(node.iter_pruned_resources())}'
            for i in rs:  # branch on resources

                # assign task -> resource
                cts = node.cts.copy()
                cts[i] += pt

                value = np.max(cts)  # evaluate objective fn for new schedule

                if (
                    value < best.value or not node.schedule
                ):  # prune sub-optimal
                    schedule = assoc(node.schedule, j, i)
                    state = hash(tuple(np.sort(cts)))

                    if state not in closed:  # prune symmetries
                        closed.add(state)  # noqa: PD005
                        rts = node.rts.copy()
                        rts[j] -= pt
                        h_value = h(R, rts, cts)
                        if h_value < best.value:
                            heappush(
                                queue,
                                Solution(
                                    schedule, cts, rts, value, h_value, depth
                                ),
                            )
                        else:
                            pruned += R ** (n - depth)
                            # TODO: pruned_h_value += R ** (n - depth)
                            pruned_h_value += 1
                    else:
                        pruned += R ** (n - depth)
                        pruned_closed += 1
                else:
                    pruned += R ** (n - depth)
                    pruned_value += 1

    stats = Stats(
        R=R,
        n=n,
        makespan=best.value,
        max_open=max_open,
        expanded=expanded,
        pruned=pruned,
        pruned_value=pruned_value,
        pruned_h_value=pruned_h_value,
        pruned_closed=pruned_closed,
        proved_optimal=proved_optimal,
        elapsed=datetime.now() - start,
    )
    return best.task_allocation, stats


def milp(
    R: int, p: Sequence[float], solver: Optional[pl.LpSolver] = None
) -> Tuple[List[int], float]:
    if R < 1 or not p:
        return [], 0.0

    n = len(p)

    problem = pl.LpProblem('Minimum_Makespan_Problem', pl.LpMinimize)

    # solve LPT to get a tight upper bound on the solution
    _, ub, _ = lpt(R, p)

    # objective: minimize makespan
    v = pl.LpVariable('v', lowBound=0, upBound=ub, cat=pl.LpContinuous)
    problem += v

    # x[i][j] = 1 iff task j is allocated to resource i
    x = pl.LpVariable.dicts('x', (range(R), range(n)), cat=pl.LpBinary)

    # makespan is the maximum completion time across resources
    for i in range(R):
        problem += v >= (p[j] * x[i][j] for j in range(n))

    # task can be assigned once to exacly one resource
    for j in range(n):
        problem += sum(x[i][j] for i in range(R)) == 1

    problem.solve(solver)

    # check solution quality
    if problem.status is not pl.LpSolutionOptimal:
        return [], 0.0

    # extract solution
    schedule = [0] * n
    for j in range(n):
        for i in range(R):
            if x[i][j].value() == 1:
                schedule[j] = i
                continue

    return schedule, pl.value(problem.objective)


def feasible(x: np.ndarray, R: int) -> bool:
    """Checks that each resource is utilized (if n >= R)"""
    return len(x) < R or len(np.unique(x)) == R


def init(R: int, n: int, init_feasible: bool = True) -> np.ndarray:
    x = np.random.randint(R, size=n)
    while init_feasible and not feasible(x, R):
        x = np.random.randint(R, size=n)
    return x


def vec_tweak(x: np.ndarray, R: int, copy: bool = True) -> np.ndarray:
    """Global mutation operator"""
    n = len(x)
    tweak_prob = 1.0 / n
    if copy:
        x = x.copy()
    for j in range(n):
        if np.random.random() < tweak_prob:
            x[j] = np.random.randint(R)
    return x


def point_tweak(x: np.ndarray, R: int, copy: bool = True) -> np.ndarray:
    """Not a global operator"""
    if copy:
        x = x.copy()
    point = np.random.randint(len(x))
    x[point] = np.random.randint(R)
    return x


def swap_tweak(x: np.ndarray, R: int, copy: bool = True) -> np.ndarray:
    """Not a global operator, also x should contain full range of R"""
    n = len(x)
    if copy:
        x = x.copy()
    i = np.random.randint(n)
    j = np.random.randint(n)
    x[i], x[j] = x[j], x[i]
    return x


def anneal(
    R: int,
    p: Sequence[float],
    tweak: Callable[[np.ndarray, int], np.ndarray] = vec_tweak,
    t0_ratio: float = 1,
    cooling: float = 0.1,
    max_iters: int = 1000,
    init_feasible: bool = True,
    penalize: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    if R < 1 or not p:
        return [], float('nan')

    if seed is not None:
        np.random.seed(seed)

    n = len(p)
    M = sum(p) if penalize and n >= R else 0

    def quality(x: np.ndarray, penalize: bool = True) -> float:
        c = np.zeros(R, np.float64)
        res = set()
        for j, i in enumerate(x):
            c[i] += p[j]
            res.add(i)  # noqa: PD005
        # add penalty m * sum(p) where m is the count of unused resources
        penalty = (R - len(res)) * M
        return cast(float, np.max(c)) + (penalty if penalize else 0)

    def temperature(i: int, best: float) -> float:
        progress = i / max_iters
        return (
            (-t0_ratio * best) * math.exp(-progress * cooling) / math.log(0.5)
        )

    s = init(R, n, init_feasible)
    m = quality(s)

    schedule = s.copy()
    makespan = quality(schedule)

    i = 0
    while i < max_iters:

        c = tweak(s, R)
        v = quality(c)

        t = temperature(i, makespan)
        accept_prob = math.exp((m - v) / t)

        if v < m or np.random.random() < accept_prob:
            s, m = c, v

        if m < makespan:
            schedule, makespan = s.copy(), m

        i += 1

    # return makespan instead of quality which might be penalized
    return schedule, quality(schedule, penalize=False)


Crossover = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
Mutation = Callable[[np.ndarray], None]


def mutation(
    tweak: Callable[[np.ndarray, int, bool], np.ndarray], R: int
) -> Mutation:
    return cast(Mutation, lambda x: tweak(x, R, False))


def npoint_crossover(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a, b = x.copy(), y.copy()
    if n < 1:
        return a, b

    c = np.random.randint(len(x), size=n)
    c.sort()

    if n == 1:
        c = first(c)
        if c:
            for i in range(c - 1):
                a[i], b[i] = b[i], a[i]
    else:
        for i, j in zip(c, c[1:]):
            if i != j:
                for k in range(i, j - 1):
                    a[k], b[k] = b[k], a[k]

    return a, b


def one_point_crossover(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return npoint_crossover(x, y, n=1)


def two_point_crossover(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return npoint_crossover(x, y, n=2)


def uniform_crossover(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    a, b, n = x.copy(), y.copy(), len(x)
    swap_prob = 1.0 / n
    for i in range(n):
        if np.random.random() < swap_prob:
            a[i], b[i] = b[i], a[i]
    return a, b


def tournament_select(fitness: np.ndarray, size: int) -> int:
    """Tournament selection"""
    pop_size = len(fitness)
    winner: int = np.random.randint(pop_size)
    for _ in range(2, size):
        i = np.random.randint(pop_size)
        if fitness[i] < fitness[winner]:
            winner = i
    return winner


def evolve(
    R: int,
    p: Sequence[float],
    mate: Crossover,
    mutate: Mutation,
    pop_size: int = 100,
    tournament_size: int = 3,
    max_iters: int = 1000,
    penalize: bool = True,
    lpt_seed: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    if R < 1 or not p:
        return [], float('nan')

    if seed is not None:
        np.random.seed(seed)

    n = len(p)
    M = sum(p) if penalize and n >= R else 0

    def fitness(x: np.ndarray, penalize: bool) -> float:
        """Compute fitness (makespan) of given individual (task allocation)"""

        c = np.zeros(R, np.float64)
        res = set()
        for j, i in enumerate(x):
            c[i] += p[j]
            res.add(i)  # noqa: PD005

        # add penalty m * sum(p) where m is the count of unused resources
        penalty = (R - len(res)) * M
        return cast(float, np.max(c)) + (penalty if penalize else 0)

    def find_best(
        pop: Sequence[np.ndarray], fit: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Find the fittest individual in population and its fitness"""
        amin = fit.argmin()
        return pop[amin], fit[amin]

    def next_offsprings(
        pop: Sequence[np.ndarray], fit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pick two parents using tournament selection and produce two offsprings
        """
        # TODO: upgrade to memetic alg by running some optimization inside
        parent1 = pop[tournament_select(fit, tournament_size)]
        parent2 = pop[tournament_select(fit, tournament_size)]

        child1, child2 = mate(parent1, parent2)

        return mutate(child1), mutate(child2)

    def generate_offsprings(
        pop: Sequence[np.ndarray], fit: np.ndarray
    ) -> Iterable[np.ndarray]:
        """Generate infinite sequence of new individuals"""
        while True:
            yield from next_offsprings(pop, fit)

    assess_fitness = np.vectorize(
        fitness,
        signature='(n)->()',
        excluded=['penalize'],
    )
    next_generation = compose(list, take(pop_size), generate_offsprings)

    # population of individuals (task allocations)
    population = [init(R, n) for _ in range(pop_size)]

    # seed initial population with LPT solution
    if lpt_seed:
        lpt_solution, _, _ = lpt(R, p)
        population[0] = np.array(lpt_solution)

    # keep track of the overall best individual
    best, best_fitness = None, None

    k = 0
    while k < max_iters:

        # assess fitness
        pop_fitness = assess_fitness(population, penalize=penalize)

        # find elite and update best individual
        elite, elite_fitness = find_best(population, pop_fitness)
        if best_fitness is None or elite_fitness < best_fitness:
            best, best_fitness = elite, elite_fitness

        # create next generation
        population = next_generation(population, pop_fitness)

        # elitism
        population[0] = elite

        k += 1

    # return makespan instead of fitness which might be penalized
    return best, fitness(best, penalize=False)


def evo_strat(
    R: int,
    p: Sequence[float],
    num_parents: int = 10,
    pop_size: int = 100,
    elitism: bool = False,
    tweak: Callable[[np.ndarray, int], np.ndarray] = vec_tweak,
    max_iters: int = 1000,
    penalize: bool = True,
    lpt_seed: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Evolutionary Strategy:
     - (num_parents,pop_size) ES if elitism is disabled (default)
     - (num_parents+pop_size) ES if elitism is enabled
    """
    assert num_parents <= pop_size and pop_size % num_parents == 0

    if R < 1 or not p:
        return [], float('nan')

    if seed is not None:
        np.random.seed(seed)

    n = len(p)
    M = sum(p) if penalize and n >= R else 0

    def neg_fit(fit: Tuple[int, float]) -> float:
        _, f = fit
        return -f

    def fitness(x: np.ndarray, penalize: bool) -> float:
        c = np.zeros(R, np.float64)
        res = set()
        for j, i in enumerate(x):
            c[i] += p[j]
            res.add(i)  # noqa: PD005
        # add penalty m * sum(p) where m is the count of unused resources
        penalty = (R - len(res)) * M
        return cast(float, np.max(c)) + (penalty if penalize else 0)

    assess_fitness = np.vectorize(
        fitness,
        signature='(n)->()',
        excluded=['penalize'],
    )

    # population of individuals (task allocations)
    population = [init(R, n) for _ in range(pop_size)]

    # seed initial population with LPT solution
    if lpt_seed:
        lpt_solution, _, _ = lpt(R, p)
        population[0] = np.array(lpt_solution)

    # keep track of the overall best individual
    best, best_fitness = None, None

    k = 0
    while k < max_iters:

        # assess fitness
        pop_fitness = assess_fitness(population, penalize=penalize)

        # find `num_parents` fittest individuals
        parents = topk(num_parents, enumerate(pop_fitness), key=neg_fit)

        # keep track of the best individual
        elite, elite_fitness = first(parents)
        if best_fitness is None or elite_fitness < best_fitness:
            best, best_fitness = population[elite], elite_fitness

        # produce new population by mutating the parents (equal proportions)
        parents = [population[parent] for parent, _ in parents]
        population = [
            tweak(parent, R)
            for _ in range(pop_size // num_parents)
            for parent in parents
        ]

        # promote parents in (pop_size + num_parents) ES
        if elitism:
            population = parents + population

        k += 1

    # return makespan instead of fitness which might be penalized
    return best, fitness(best, penalize=False)


def hill_climb(
    quality: Callable[[np.ndarray], float],
    tweak: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    schedule: np.ndarray,
) -> Tuple[np.ndarray, float]:
    makespan = quality(schedule)

    for _ in range(max_iters):

        candidate = tweak(schedule)
        candidate_makespan = quality(candidate)

        if candidate_makespan < makespan:
            schedule, makespan = candidate.copy(), candidate_makespan

    return schedule, makespan


def grasp(
    R: int,
    p: Sequence[float],
    cc_ratio: float = 0.5,
    tweak: Callable[[np.ndarray, int], np.ndarray] = vec_tweak,
    max_iters: int = 1000,
    hc_iters: int = 10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    :param cc_ratio: % of best feasible candidate components to pick from
    :param hc_iters: # iterations to hill-climb candidate solution each epoch
    """
    if R < 1 or not p:
        return np.array([]), float('nan')

    if seed is not None:
        np.random.seed(seed)

    n = len(p)
    components = frozenset((i, j) for i in range(R) for j in range(n))

    def processing_time(assignment: Tuple[int, int]) -> float:
        _, j = assignment
        return p[j]

    def next_candidate() -> np.ndarray:
        # (partial) schedule is represented as a mapping `{task: resource}`
        schedule: Dict[int, int] = {}
        # repeat until a complete feasible schedule is found
        while len(schedule) < n:
            # if i <- j exists =>  all (*, j) are infeasible
            feasible = sorted(
                ((i, j) for i, j in components if j not in schedule),
                key=processing_time,
            )

            if not feasible:
                schedule = {}

            # components sorted by `p[j']` desc => pick random one from k best
            kbest = int(max(cc_ratio * len(feasible), 1))
            i, j = feasible[np.random.randint(kbest)]
            schedule[j] = i

        candidate = np.zeros(n, np.uint16)
        for j, i in schedule.items():
            candidate[j] = i
        return candidate

    def quality(x: np.ndarray) -> float:
        c = np.zeros(R, np.float64)
        for j, i in enumerate(x):
            c[i] += p[j]
        return cast(float, np.max(c))

    def hc_tweak(x: np.ndarray) -> np.ndarray:
        return tweak(x, R)

    best: Optional[np.ndarray] = None
    best_makespan = None

    for _ in range(max_iters):
        schedule = next_candidate()
        schedule, makespan = hill_climb(quality, hc_tweak, hc_iters, schedule)
        if best_makespan is None or makespan < best_makespan:
            best, best_makespan = schedule.copy(), makespan

    assert best is not None and best_makespan is not None
    return best, best_makespan
