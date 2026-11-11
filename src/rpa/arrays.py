"""Small numpy helpers shared across modules."""

import numpy as np


def ranges(counts: np.ndarray) -> np.ndarray:
    """Concatenated arange(c) for each c in counts: [2, 3] -> [0, 1, 0, 1, 2]."""
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    return np.arange(int(counts.sum())) - np.repeat(starts, counts)
