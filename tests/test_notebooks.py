"""Notebook syntax, portable imports, and absence of artifact-writing cells."""

import ast
import json
from pathlib import Path


def test_notebooks_are_portable_and_have_no_saved_outputs():
    paths = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
    assert {p.stem for p in paths} == {
        "fig1",
        "fig2",
        "fig3",
        "fig4",
        "fig5",
        "figS1",
        "figS2",
        "figS3",
        "figS4",
    }
    for path in paths:
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None and cell["outputs"] == []
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {
                        "savefig",
                        "save",
                        "savez",
                        "savez_compressed",
                        "save_mesh",
                        "write_text",
                        "mkdir",
                    }
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] in {
                        "pathlib",
                        "jax",
                        "numpy",
                        "scipy",
                        "matplotlib",
                        "mpl_toolkits",
                        "cvm3d",
                    }


def test_first_cells_find_bundled_package_without_pythonpath(tmp_path):
    """Exercise actual first cells with no cwd/PYTHONPATH import assistance."""
    import shutil
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "renamed-release"
    shutil.copytree(
        root, copied, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    for directory, expected_root in (
        (root / "notebooks", root),
        (root, root),
        (root.parent, root),
        (copied / "notebooks", copied),
    ):
        # Reset cvm3d imports between notebooks to exercise each independently.
        script = """
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for notebook in sorted((root / 'notebooks').glob('*.ipynb')):
    for name in list(sys.modules):
        if name == 'cvm3d' or name.startswith('cvm3d.'):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if p != str(root)]
    cell = next(c for c in json.loads(notebook.read_text())['cells'] if c['cell_type'] == 'code')
    exec(compile(''.join(cell['source']), str(notebook), 'exec'), {})
    import cvm3d
    assert Path(cvm3d.__file__).resolve().parent == root / 'cvm3d'
"""
        subprocess.run(
            [sys.executable, "-I", "-c", script, str(expected_root)],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
