"""Simple shear: periodic images, independent polygon energy, linear response."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from cvm3d import make_regular_hexagonal_mesh
from cvm3d.energy import energy_from_tables
from cvm3d.geometry import face_vertices_unfolded
from cvm3d.regular import equilibrium_side_length
from cvm3d.rheology import complex_modulus


def mesh():
    l = equilibrium_side_length(gamma_b=-7.0, gamma_c=-8.0, sigma=3.0, kappa=0.1)
    return make_regular_hexagonal_mesh(
        5, 4, l, 1.0, gamma_b=-7.0, gamma_c=-8.0, sigma=3.0
    )


def energy(m, vertices, box, half_edges=None):
    return energy_from_tables(
        vertices,
        m.half_edges if half_edges is None else half_edges,
        m.faces,
        m.vertices_params,
        m.half_edges_params,
        m.faces_params,
        width=m.width,
        height=m.height,
        total_volume=m.total_volume,
        edge_star_strength=0.1,
        max_edges=m.max_edges,
        box_matrix=box,
        target_cell_volumes=1.0,
        volume_penalty=1e5,
    )


def test_finite_shear_energy_and_periodic_images():
    m = mesh()
    # Independent NumPy polygons, unfolded BEFORE deforming the periodic cell.
    polygons = []
    for f in range(m.nb_faces):
        p, valid = face_vertices_unfolded(
            f,
            m.vertices,
            m.half_edges,
            m.faces,
            m.width,
            m.height,
            max_edges=m.max_edges,
        )
        polygons.append(np.asarray(p)[np.asarray(valid)])
    for gamma in (-0.3, 0.0, 0.4):
        F = np.array([[1.0, gamma], [0.0, 1.0]])
        box = F @ np.diag([m.width, m.height])
        points = [p @ F.T for p in polygons]
        areas = np.array(
            [
                abs(
                    np.sum(
                        p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)
                    )
                )
                / 2
                for p in points
            ]
        )
        h = m.total_volume / (areas.sum() + 1e-12)
        expected = 0.0
        for p, area in zip(points, areas):
            prev = np.roll(p, 1, axis=0) - p
            nxt = np.roll(p, -1, axis=0) - p
            stars = (
                prev[:, :, None] * prev[:, None, :] + nxt[:, :, None] * nxt[:, None, :]
            )
            perimeter = np.linalg.norm(nxt, axis=1).sum()
            expected += -7 * area - 4 * h * perimeter + 3 * perimeter
            expected += (
                0.1
                * 2
                / 3
                * np.sum(np.trace(np.linalg.inv(stars), axis1=1, axis2=2) + 1 / h**2)
            )
        expected += 0.5e5 * np.sum((areas * h - 1) ** 2)
        vertices = m.vertices @ jnp.asarray(F).T
        np.testing.assert_allclose(
            energy(m, vertices, box), expected, rtol=1e-11, atol=1e-9
        )
        # Move one vertex to another image and compensate integer edge offsets.
        shift = np.array([1, -1])
        he = np.array(m.half_edges)
        he[:, 6:8] += (he[:, 3:4] == 0) * shift - (he[:, 4:5] == 0) * shift
        moved = vertices.at[0].add(jnp.asarray(box @ shift))
        np.testing.assert_allclose(
            energy(m, moved, box, jnp.asarray(he)), expected, rtol=1e-11, atol=1e-9
        )


def test_regular_linear_response_shear_equivalence():
    m = mesh()
    opts = dict(edge_star_strength=0.1, target_cell_volumes=1.0, volume_penalty=1e5)
    pure = complex_modulus(m, [0.0, 1.0, 1e4], mode="shear", **opts)
    simple = complex_modulus(m, [0.0, 1.0, 1e4], mode="simple_shear", **opts)
    np.testing.assert_allclose(
        simple.complex_modulus, pure.complex_modulus, rtol=2e-6, atol=1e-6
    )

    # Independent finite difference of the affine energy checks engineering strain normalization.
    def e(g):
        F = jnp.array([[1.0, g], [0.0, 1.0]])
        return float(
            energy(m, m.vertices @ F.T, F @ jnp.diag(jnp.array([m.width, m.height])))
        )

    step = 1e-3
    stiffness = (
        -e(2 * step) + 16 * e(step) - 30 * e(0.0) + 16 * e(-step) - e(-2 * step)
    ) / (12 * step**2)
    np.testing.assert_allclose(
        simple.affine_modulus, stiffness / (m.width * m.height), rtol=1e-6
    )
