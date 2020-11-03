import json
import os
import pathlib
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

from alg import Heuristic, Stats, bnb

_root_path = pathlib.Path(__file__).parent.parent.absolute()
_data_path = os.path.join(_root_path, 'data')


def load_sample(n: int) -> List[float]:
    file_path = os.path.join(_data_path, f'sample{n}.json')
    with open(file_path, 'rb') as file:
        return json.load(file)  # type: ignore


SAMPLE_INSTANCE: Sequence[float] = load_sample(1)


def gantt(R: int, p: Sequence[float], s: Sequence[int]) -> pd.DataFrame:
    alloc = {i: [t for t, r in enumerate(s) if r == i] for i in range(R)}

    g = []
    for i, ts in alloc.items():
        c = 0.0
        for t in ts:
            g.append(
                {'task': t, 'start': c, 'finish': c + p[t], 'resource': i}
            )
            c += p[t]

    return pd.DataFrame(g)


def suboptimal_instance(R: int) -> List[float]:
    p: List[float] = []
    pt = 2 * R - 1
    while pt >= R:
        p.append(pt)
        p.append(pt)
        pt -= 1
    p.append(R)
    return p


def generate_subopt_instances(m: int) -> Iterable[Tuple[int, List[float]]]:
    for R in range(m):
        yield R, suboptimal_instance(R)


def random_instance(n: int, high: float) -> List[float]:
    return list(np.random.randint(1, high, size=n))


def generate_random_instances(
    m: int, high: float = 10
) -> Iterable[Tuple[int, List[float]]]:
    for n in range(m):
        yield n, random_instance(n, high)


T = TypeVar('T', covariant=True)


def evaluate(
    f: Callable[[Stats], T], R: int, p: Sequence[float], h: Heuristic
) -> T:
    _, stats = bnb(R, p, h)
    return f(stats)
