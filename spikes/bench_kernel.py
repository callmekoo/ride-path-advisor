"""Spike S2: is the kernel fast enough on the real regional graph?

Pass criteria from the design: a full single-source search under 400 ms and a
~50 km point-to-point search under 30 ms (median).

    uv run python spikes/bench_kernel.py [--graph data/graph/current]
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

import rpa_kernel
from rpa import config
from rpa.graph.snap import Snapper
from rpa.graph.store import load_graph
from rpa.routing.cost import Prefs, arc_weights

# Yekaterinburg city centre: the benchmark searches start around here because
# that is where the graph is densest.
CENTRE = (60.6, 56.84)


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=config.current_graph_dir())
    parser.add_argument("--pairs", type=int, default=200)
    args = parser.parse_args()

    (graph, _), load_ms = timed(lambda: load_graph(args.graph))
    kernel, kernel_ms = timed(lambda: rpa_kernel.Graph(graph.edge_u, graph.edge_v, graph.node_lonlat))
    weights, weights_ms = timed(lambda: arc_weights(graph, Prefs()))
    print(f"graph: {graph.n_nodes:,} nodes, {graph.n_edges:,} edges")
    print(f"load {load_ms:.0f} ms, kernel build {kernel_ms:.0f} ms, arc weights {weights_ms:.0f} ms")

    length = np.ascontiguousarray(graph.edge_len_m)
    centre, _ = Snapper(graph).nearest_node(*CENTRE)

    sssp_ms = []
    for _ in range(5):
        (cost, metres, _), ms = timed(lambda: kernel.sssp(centre, weights, length))
        sssp_ms.append(ms)
    reached = np.isfinite(cost)
    print(f"full sssp: median {statistics.median(sssp_ms):.0f} ms, reaches {reached.mean():.1%} of nodes")

    _, ms = timed(lambda: kernel.sssp(centre, weights, length, max_cost=60_000 * 1.5))
    print(f"sssp limited to ~60 km of cost-distance: {ms:.0f} ms")

    rng = np.random.default_rng(0)
    for low_km, high_km in [(5, 15), (40, 60), (120, 180)]:
        candidates = np.flatnonzero(reached & (metres >= low_km * 1000) & (metres <= high_km * 1000))
        if len(candidates) == 0:
            print(f"route {low_km}-{high_km} km: no nodes at this distance")
            continue
        targets = rng.choice(candidates, size=min(args.pairs, len(candidates)), replace=False)
        times, arcs_total = [], 0
        for target in targets:
            found, ms = timed(lambda: kernel.route(centre, int(target), weights))
            times.append(ms)
            arcs_total += len(found[0])
        times.sort()
        print(
            f"route {low_km}-{high_km} km: median {statistics.median(times):.1f} ms, "
            f"p95 {times[int(len(times) * 0.95) - 1]:.1f} ms, "
            f"mean {arcs_total / len(times):.0f} arcs ({len(times)} pairs)"
        )


if __name__ == "__main__":
    main()
