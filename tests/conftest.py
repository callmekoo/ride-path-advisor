import pytest
from helpers import MICRO

from rpa.graph.store import GraphData
from rpa.osm.build import build_graph


@pytest.fixture(scope="session")
def micro() -> GraphData:
    """The graph of tests/fixtures/micro.opl. Treat as read-only."""
    return build_graph(MICRO)
