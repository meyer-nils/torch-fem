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
