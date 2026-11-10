import pytest
from helpers import MICRO

from rpa.graph.store import GraphData
from rpa.osm.build import build_graph


@pytest.fixture(scope="session")
def micro() -> GraphData:
    """The graph of tests/fixtures/micro.opl. Treat as read-only."""
    return build_graph(MICRO)


@pytest.fixture(scope="session")
def grid() -> GraphData:
    """A 21 x 21 street grid, 500 m apart (10 km x 10 km). Treat as read-only."""
    from helpers import grid_graph

    return grid_graph()


@pytest.fixture(scope="session")
def grid_engine(grid):
    from rpa.routing.engine import Engine

    return Engine(grid)
