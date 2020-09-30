from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import plotly.express as px

def gantt(R: int, p: Sequence[float], s: Sequence[int]) -> pd.DataFrame:
    alloc = {
        i: [t for t, r in enumerate(s) if r == i]
        for i in range(R)
    }
    
    g = []
    for i, ts in alloc.items():
        c = 0
        for t in ts:
            g.append({'task': t, 'start': c, 'finish': c + p[t], 'resource': i})
            c += p[t]

    return pd.DataFrame(g)


def gantt_plot_convert(df: pd.DataFrame, d: Optional[date] = None) -> pd.DataFrame:
    start = date.today() if d is None else d
    df_copy = df.copy()
    df_copy['task'] = df_copy['task'].apply(lambda t: f'T-{t}')
    df_copy['start'] = df_copy['start'].apply(lambda t: start + timedelta(days=t))
    df_copy['finish'] = df_copy['finish'].apply(lambda t: start + timedelta(days=t))
    df_copy['resource'] = df_copy['resource'].apply(lambda r: f'R-{r}')
    return df_copy


def show_gantt(df: pd.DataFrame) -> None:
    df = gantt_plot_convert(df)
    fig = px.timeline(df, x_start='start', x_end='finish', y='resource', color='resource')
    fig.show()

    
def suboptimal_instance(R: int) -> List[float]:
    p = []
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


def generate_random_instances(m: int, high: float = 10) -> Iterable[Tuple[int, List[float]]]:
    for n in range(m):
        yield n, random_instance(n, high)
