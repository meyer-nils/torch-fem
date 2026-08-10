from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet
from testbook import testbook

# Slow: these execute the example notebooks. Skip with `-m "not notebook"`.
pytestmark = pytest.mark.notebook

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# Notebooks that are too slow to execute on every test run.
SKIP = {
    "optimization/solid/bracket.ipynb": "runs for more than ten minutes",
    "optimization/solid/topology_thermal.ipynb": "runs for more than two minutes",
    "optimization/planar/fiber_patch_placement.ipynb": "runs for about two minutes",
    "optimization/planar/void_identification.ipynb": "runs for about three minutes",
}


def _notebooks() -> list[ParameterSet]:
    """Every example notebook, identified by its path below `examples/`."""
    params = []
    for path in sorted(EXAMPLES.rglob("*.ipynb")):
        if any(part.startswith(".") for part in path.parts):
            continue
        name = path.relative_to(EXAMPLES).as_posix()
        reason = SKIP.get(name)
        marks = [pytest.mark.skip(reason=reason)] if reason else []
        params.append(pytest.param(path, id=name, marks=marks))
    return params


@pytest.mark.parametrize("notebook", _notebooks())
def test_notebook(notebook):
    with testbook(notebook, timeout=600) as tb:
        tb.execute()
