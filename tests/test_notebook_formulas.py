"""Check figure formulas against the implemented corner energy and derivatives."""

import ast
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from cvm3d import make_regular_hexagonal_mesh
from cvm3d.regular import equilibrium_side_length


def formulas(fig):
    path = Path(__file__).resolve().parents[1] / f"notebooks/fig{fig}.ipynb"
    cells = json.loads(path.read_text())["cells"]
    namespace = dict(
        np=np, SQRT3=np.sqrt(3.0), KAPPA=0.05, V0=1.0, minimize_scalar=minimize_scalar
    )
    wanted = {
        "energy",
        "energy_isotropic",
        "energy_deformed",
        "bulk_modulus",
        "shear_modulus",
        "denergy_dl",
        "d2energy_dl2",
    }
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), str(path), "exec"
                    ),
                    namespace,
                )
    return namespace


def test_fig2_finite_strain_energy_matches_microscopic_prisms():
    ns = formulas(2)
    l = 0.8
    alpha = 1.03
    beta = 0.98
    params = (-1.0, -2.0, 3.0)
    mesh = make_regular_hexagonal_mesh(
        5, 4, l, 1.0, gamma_b=params[0], gamma_c=params[1], sigma=params[2]
    )
    mesh.vertices = mesh.vertices * jnp.array([alpha, beta])
    mesh.width *= alpha
    mesh.height *= beta
    np.testing.assert_allclose(
        mesh.energy(0.05) / mesh.nb_faces,
        ns["energy_deformed"](beta, params, alpha, l),
        rtol=1e-10,
    )


def test_fig2_fig3_moduli_match_energy_curvature():
    params = (-1.0, -2.0, 3.0)
    l = equilibrium_side_length(
        gamma_b=params[0], gamma_c=params[1], sigma=params[2], kappa=0.05
    )
    ns2 = formulas(2)
    ns3 = formulas(3)
    area = 1.5 * np.sqrt(3) * l * l

    def second(f, h=1e-3):
        return (-f(2 * h) + 16 * f(h) - 30 * f(0) + 16 * f(-h) - f(-2 * h)) / (
            12 * h * h
        )

    K = second(lambda e: ns2["energy_deformed"](1 + e, params, 1 + e, l)) / (4 * area)
    G = second(lambda e: ns2["energy_deformed"](1 / (1 + e), params, 1 + e, l)) / (
        4 * area
    )
    for ns in (ns2, ns3):
        np.testing.assert_allclose(ns["bulk_modulus"](l, params), K, rtol=1e-7)
        np.testing.assert_allclose(ns["shear_modulus"](l, params), G, rtol=1e-7)
