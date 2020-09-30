from datetime import date, datetime, timedelta
from typing import List, Sequence, Tuple

import numpy as np


def lpt(R: int, p: Sequence[float]) -> Tuple[List[int], float, float]:
    if R < 1:
        return [], 0.0, float('nan')
    pts = sorted(enumerate(p), key=lambda x: x[1], reverse=True)
    cts = np.zeros(R, np.float64)
    hist = [0] * R
    
    schedule = [0] * len(pts)
    for t, pt in pts:
        i: int = (cts + pt).argmin()
        cts[i] += pt
        hist[i] += 1
        schedule[t] = i
        
    k = max(hist)
    r = 1 + (1 / k) + (1 / (k * R)) if k > 0 else 1.0
    return schedule, np.max(cts), r
