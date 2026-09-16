import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from cvm3d import make_regular_hexagonal_mesh
from cvm3d.regular import regular_cell_energy, equilibrium_side_length
from cvm3d.z_shear import (
    ShearMesh,
    CellParameters,
    regular_energy,
    affine_shear_energy_difference,
    regular_energy_derivative,
)


def mesh(l=0.8):
    return make_regular_hexagonal_mesh(
        5, 4, l, 1.0, gamma_b=0.0, gamma_c=-8.0, sigma=5.0
    )


def test_regular_energy_and_equilibrium():
    params = dict(gamma_b=0.0, gamma_c=-8.0, sigma=5.0, kappa=0.05)
    p = CellParameters(**params)
    for l in (0.25, 0.8):
        m = mesh(l)
        assert m.vertices.dtype == jnp.float64
        expected = regular_cell_energy(l, **params)
        np.testing.assert_allclose(m.energy(0.05) / m.nb_faces, expected, rtol=1e-10)
        np.testing.assert_allclose(regular_energy(l, p), expected, rtol=1e-12)
    l = equilibrium_side_length(**params)
    assert abs(regular_energy_derivative(l, p)) < 1e-3


def test_optical_mode_area_and_star_curvature():
    m = mesh()
    he = np.asarray(m.half_edges)
    colors = np.zeros(m.nb_vertices)
    colors[0] = 1
    pending = [0]
    while pending:
        v = pending.pop()
        for w in he[he[:, 3] == v, 4]:
            if colors[w] == 0:
                colors[w] = -colors[v]
                pending.append(w)
            assert colors[w] == -colors[v]
    direction = jnp.asarray(np.column_stack((colors / 2, np.zeros_like(colors))))

    def components(t):
        moved = m.copy()
        moved.vertices = m.vertices + t * direction
        return moved.energy_components(
            edge_star_strength=0.05, target_cell_volumes=1.0, volume_penalty=1e3
        )

    a = jax.grad(jax.grad(lambda t: components(t)["edge_star"] / m.nb_faces))(0.0) / 2
    np.testing.assert_allclose(a, 160 * 0.05 / (3 * 0.8**4), rtol=1e-10)
    np.testing.assert_allclose(components(0.01)["cell_volumes"], 1.0, rtol=1e-10)


def test_z_shear_matches_prism_and_affine_formula():
    l = 0.8
    p = CellParameters(gamma_b=0.0, gamma_c=-8.0, sigma=5.0, kappa=0.05)
    m = ShearMesh.regular_hexagonal(5, 4, l)
    zero = m.zero_displacements()
    baseline = m.energy_components(zero, m.edge_aligned_shear(0.0), p)
    np.testing.assert_allclose(
        baseline["physical"] / m.cell_count, mesh(l).energy(0.05) / 20, rtol=1e-10
    )
    for q in (0.02, 0.2):
        actual = (
            m.energy_components(zero, m.edge_aligned_shear(q), p)["physical"]
            / m.cell_count
        )
        np.testing.assert_allclose(
            actual - baseline["physical"] / m.cell_count,
            affine_shear_energy_difference(l, q, p),
            rtol=1e-9,
            atol=1e-11,
        )


def test_soft_targets_are_initial_volumes_and_fixed():
    m = mesh()
    m.vertices = m.vertices.at[0].add(jnp.array([0.01, -0.02]))
    targets = np.array(m.energy_components()["cell_volumes"], copy=True)
    opts = dict(target_cell_volumes=targets, volume_penalty=1e3)
    assert float(m.energy_components(**opts)["volume_penalty"]) < 1e-20
    m.vertices = m.vertices.at[0].add(jnp.array([0.01, 0.0]))
    c = m.energy_components(**opts)
    np.testing.assert_allclose(
        c["volume_penalty"],
        500 * np.sum((np.asarray(c["cell_volumes"]) / targets - 1) ** 2),
    )
