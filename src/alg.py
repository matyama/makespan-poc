import math

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from heapq import heappop, heappush
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pulp as pl

from cytoolz.dicttoolz import assoc
from cytoolz.itertoolz import first, second, unique


def lpt(R: int, p: Sequence[float]) -> Tuple[List[int], float, float]:
    if R < 1:
        return [], 0.0, float('nan')
    pts = sorted(enumerate(p), key=second, reverse=True)
    schedule, _, makespan, r = _lpt(R, pts)
    return schedule, makespan, r


def _lpt(R: int, p: Sequence[Tuple[int, float]]) -> Tuple[List[int], np.ndarray, float, float]:
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
        return cls(schedule=dict(enumerate(schedule)), cts=cts, rts=np.zeros(n, np.float64), value=value, h_value=value, depth=n)

    @classmethod
    def init(cls, R: int, p: Sequence[float], h_value: float) -> 'Solution':
        return cls(schedule={}, cts=np.zeros(R, np.float64), rts=np.array(p), value=0.0, h_value=h_value, depth=0)
    

def h1(R: int, rts: np.ndarray, cts: np.ndarray) -> float:
    cts = cts.copy()
    time_left = np.sum(rts)

    while time_left > 0:

        if np.all(cts == cts[0]):
            # distribute tasks uniformly if all completion times are the same
            return cts[0] + (time_left / R)

        a_min = np.argmin(cts)
        a_max = np.argmax(cts)
        diff = cts[a_max] - cts[a_min]

        cts[a_min] += diff
        time_left -= diff

    return np.max(cts)


def h2(R: int, rts: np.ndarray, cts: np.ndarray) -> float:
    return max(np.max(rts), (np.sum(cts) + np.sum(rts)) / R)


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
        return self.R ** self.n
        
    @property
    def exhaustiveness_ratio(self) -> float:
        return self.expanded / self.space_size if self.space_size > 0 else 1
        
    @property
    def pruned_total(self) -> int:
        return self.pruned_value + self.pruned_h_value + self.pruned_closed


def bnb(R: int, p: Sequence[float], h: Heuristic, limit: Optional[timedelta] = None) -> Tuple[List[int], Stats]:
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
        return {}, stats

    closed = set()

    best = Solution.lpt(R, pts)

    queue: List[Solution] = []
    heappush(queue, Solution.init(R, p, best.value))
    total_time = sum(pt for _, pt in pts)

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

            # heuristic: select the biggest task that restricts the space the most
            j, pt = next((j, pt) for j, pt in pts if j not in node.schedule)

            # TODO: check - compare with node.iter_pruned_resources()
            _, rs = np.unique(node.cts, return_index=True)
            assert frozenset(rs) == frozenset(node.iter_pruned_resources()), f'{rs} != {list(node.iter_pruned_resources())}'
            for i in rs:  # branch on resources

                # assign task -> resource
                cts = node.cts.copy()
                cts[i] += pt

                value = np.max(cts)  # evaluate objective fn for new schedule

                if value < best.value or not node.schedule:  # prune sub-optimal
                    schedule = assoc(node.schedule, j, i)
                    state = hash(tuple(np.sort(cts)))

                    if state not in closed:  # prune symmetries
                        closed.add(state)
                        rts = node.rts.copy()
                        rts[j] -= pt
                        h_value = h(R, rts, cts)
                        if h_value < best.value:
                            heappush(queue, Solution(schedule, cts, rts, value, h_value, depth))
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


def milp(R: int, p: Sequence[float], solver: Optional[pl.LpSolver] = None) -> Tuple[List[int], float]:
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


def vec_tweak(x, R):
    """Global mutation operator"""
    n = len(x)
    tweak_prob = 1. / n
    x = x.copy()
    for j in range(n):
        if np.random.random() < tweak_prob:
            x[j] = np.random.randint(R)
    return x


def point_tweak(x, R):
    """Not a global operator"""
    x = x.copy()
    point = np.random.randint(len(x))
    x[point] = np.random.randint(R)
    return x


def swap_tweak(x, R):
    """Not a global operator, also x should contain full range of R"""
    n = len(x)
    x = x.copy()
    i = np.random.randint(n)
    j = np.random.randint(n)
    x[i], x[j] = x[j], x[i]
    return x


def sa(R, p, tweak=vec_tweak, t0_ratio=1, cooling=0.1, max_iters=1000, init_feasible=True, seed=None):
    if R < 1 or not p:
        return None, 0
    
    if seed is not None:
        np.random.seed(seed)
    
    def quality(x):
        c = np.zeros(R, np.float64)
        for j, i in enumerate(x):
            c[i] += p[j]
        return np.max(c)
    
    def temperature(i, best):
        progress = i / max_iters
        return (-t0_ratio * best) * math.exp(-progress * cooling) / math.log(0.5)
    
    def feasible(x, R):
        """Checks that each resource is utilized (if n >= R)"""
        return len(x) < R or len(np.unique(x)) == R
    
    def init(R, n):
        x = np.random.randint(R, size=n)
        while init_feasible and not feasible(x, R):
            x = np.random.randint(R, size=n)
        return x
    
    n = len(p)
    s = init(R, n)
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
            
    return schedule, makespan
