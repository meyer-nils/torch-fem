from testbook import testbook


@testbook("examples/basic/planar.ipynb", execute=True)
def test_planar_notebook(tb):
    tb.execute()


@testbook("examples/basic/shell.ipynb", execute=True)
def test_shell_notebook(tb):
    tb.execute()


@testbook("examples/basic/truss.ipynb", execute=True)
def test_truss_notebook(tb):
    tb.execute()


@testbook("examples/basic/solid.ipynb", execute=True)
def test_solid_notebook(tb):
    tb.execute()


@testbook("examples/optimization/topology_optimization.ipynb", execute=True, timeout=90)
def test_topology_optimization_notebook(tb):
    tb.execute()


@testbook("examples/optimization/shape_optimization_planar.ipynb", execute=True)
def test_shape_optimization_planar_notebook(tb):
    tb.execute()


@testbook("examples/optimization/orientation_optimization_planar.ipynb", execute=True)
def test_orientation_optimization_planar_notebook(tb):
    tb.execute()


@testbook("examples/optimization/orientation_optimization_shell.ipynb", execute=True)
def test_orientation_optimization_shell_notebook(tb):
    tb.execute()
