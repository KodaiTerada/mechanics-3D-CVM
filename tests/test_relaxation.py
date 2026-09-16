"""Convergence and failure semantics for the hybrid fixed-topology finish."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from cvm3d import make_regular_hexagonal_mesh
from cvm3d.relaxation import relax_fixed_topology


def stiff_problem():
    mesh = make_regular_hexagonal_mesh(
        5, 4, 0.8, 1.0, gamma_b=0.0, gamma_c=0.0, sigma=1.0
    )
    reference = mesh.vertices.copy()
    mesh.vertices = mesh.vertices.at[0].add(jnp.array([1e-10, 1e-4]))

    def energy(vertices, *_):
        displacement = vertices - reference
        displacement -= displacement.mean(axis=0)
        # A large offset makes energy changes unresolvable before force
        # convergence. The true minimum and its force are known exactly.
        return 1e7 + jnp.sum(
            5e5 * displacement[:, 0] ** 2 + 0.5 * displacement[:, 1] ** 2
        )

    return mesh, energy


def test_hessian_polish_reaches_force_tolerance_at_energy_precision_limit():
    mesh, energy = stiff_problem()
    original_vertices = np.asarray(mesh.vertices).copy()
    result = relax_fixed_topology(
        mesh,
        energy,
        adam_iterations=0,
        bfgs_max_iterations=0,
        max_retries=0,
        gradient_tolerance=5e-7,
    )
    assert result.converged
    assert result.gradient_max_abs <= 5e-7
    assert any("Hessian polish" in stage for stage in result.stages)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(mesh.faces, result.mesh.faces)
    np.testing.assert_array_equal(mesh.half_edges[:, :6], result.mesh.half_edges[:, :6])


def test_exhausted_iterations_do_not_report_energy_stagnation_as_convergence():
    mesh, energy = stiff_problem()
    result = relax_fixed_topology(
        mesh,
        energy,
        adam_iterations=0,
        bfgs_max_iterations=0,
        newton_iterations=0,
        max_retries=0,
        gradient_tolerance=5e-7,
    )
    assert not result.converged
    assert result.gradient_max_abs > 5e-7


def test_nonfinite_initial_state_is_rejected():
    mesh, energy = stiff_problem()
    mesh.vertices = mesh.vertices.at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="Initial geometry or energy is invalid"):
        relax_fixed_topology(mesh, energy)
