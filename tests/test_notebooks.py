from testbook import testbook


@testbook("examples/basic/planar/cantilever.ipynb", execute=True)
def test_planar_cantilever_notebook(tb):
    tb.execute()


@testbook("examples/basic/planar/fillet.ipynb", execute=True)
def test_planar_fillet_notebook(tb):
    tb.execute()


@testbook("examples/basic/planar/minimal.ipynb", execute=True)
def test_planar_minimal_notebook(tb):
    tb.execute()


@testbook("examples/basic/planar/plasticity.ipynb", execute=True)
def test_planar_plasticity_notebook(tb):
    tb.execute()


@testbook("examples/basic/shell/cantilever.ipynb", execute=True)
def test_shell_cantilever_notebook(tb):
    tb.execute()


@testbook("examples/basic/shell/plate.ipynb", execute=True)
def test_shell_plate_notebook(tb):
    tb.execute()


@testbook("examples/basic/truss/elasticity_2D.ipynb", execute=True)
def test_truss_elasticity2d_notebook(tb):
    tb.execute()


@testbook("examples/basic/truss/elasticity_3D.ipynb", execute=True)
def test_truss_elasticity3d_notebook(tb):
    tb.execute()


@testbook("examples/basic/truss/plasticity_2D.ipynb", execute=True)
def test_truss_plasticity2d_notebook(tb):
    tb.execute()


@testbook("examples/basic/solid/cubes.ipynb", execute=True)
def test_solid_cubes_notebook(tb):
    tb.execute()


@testbook("examples/basic/solid/inelasticity.ipynb", execute=True)
def test_solid_inelasticity_notebook(tb):
    tb.execute()


@testbook("examples/basic/solid/plasticity.ipynb", execute=True)
def test_solid_plasticity_notebook(tb):
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
