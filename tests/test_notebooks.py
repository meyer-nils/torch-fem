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


@testbook("examples/basic/solid/gyroid.ipynb", execute=True, timeout=90)
def test_solid_gyroid_notebook(tb):
    tb.execute()


@testbook("examples/basic/solid/inelasticity.ipynb", execute=True)
def test_solid_inelasticity_notebook(tb):
    tb.execute()


@testbook("examples/basic/solid/finite_strain.ipynb", execute=True)
def test_solid_finite_strain_notebook(tb):
    tb.execute()


@testbook("examples/basic/solid/plasticity.ipynb", execute=True)
def test_solid_plasticity_notebook(tb):
    tb.execute()


@testbook("examples/optimization/planar/topology.ipynb", execute=True)
def test_planar_topology_optimization_notebook(tb):
    tb.execute()


@testbook("examples/optimization/solid/topology.ipynb", execute=True, timeout=180)
def test_solid_topology_optimization_notebook(tb):
    tb.execute()


@testbook("examples/optimization/planar/shape.ipynb", execute=True)
def test_planar_shape_optimization_notebook(tb):
    tb.execute()


@testbook("examples/optimization/planar/orientation.ipynb", execute=True)
def test_planar_orientation_optimization_notebook(tb):
    tb.execute()


@testbook("examples/optimization/shell/orientation.ipynb", execute=True)
def test_shell_orientation_optimization_notebook(tb):
    tb.execute()
