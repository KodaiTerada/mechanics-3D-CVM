"""Check the manuscript's strain normalization and modal representation."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from cvm3d import complex_modulus, make_regular_hexagonal_mesh
from cvm3d.regular import equilibrium_side_length
from cvm3d.experiments import Parameters, volume_targets


def test_manuscript_affine_limits_and_modal_sum():
    p = Parameters()
    l = equilibrium_side_length(
        gamma_b=p.gamma_b, gamma_c=p.gamma_c, sigma=p.sigma, kappa=p.kappa
    )
    mesh = make_regular_hexagonal_mesh(
        5, 8, l, 1.0, gamma_b=p.gamma_b, gamma_c=p.gamma_c, sigma=p.sigma
    )
    np.testing.assert_allclose([mesh.width, mesh.height], [5 * np.sqrt(3) * l, 12 * l])
    targets = volume_targets(mesh, p.kappa)
    K = (
        p.gamma_b / 2
        + 2 * p.gamma_c / (9 * l**3)
        + 32 * np.sqrt(3) * p.kappa / (9 * l**4)
        + 18 * np.sqrt(3) * p.kappa * l**2
    )
    G = (
        p.gamma_c / (6 * l**3)
        + np.sqrt(3) * p.sigma / (2 * l)
        + 64 * np.sqrt(3) * p.kappa / (27 * l**4)
    )
    for mode, affine in [("bulk", K), ("simple_shear", G)]:
        response = complex_modulus(
            mesh,
            [0.0, 1.0, 1e4, 1e12],
            mode=mode,
            edge_star_strength=p.kappa,
            target_cell_volumes=targets,
            volume_penalty=1e4,
            gradient_tolerance=1e-5,
        )
        np.testing.assert_allclose(response.affine_modulus, affine, rtol=1e-6)
        active = response.hessian_eigenvalues > 1e-5
        h, xi = response.hessian_eigenvalues[active], response.strain_couplings[active]
        delta = xi**2 / (mesh.width * mesh.height * h)
        explicit = affine - np.sum(
            delta[:, None]
            / (1 + 1j * response.angular_frequencies[None, :] / h[:, None]),
            axis=0,
        )
        np.testing.assert_allclose(
            response.complex_modulus, explicit, rtol=2e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            delta.sum(), response.affine_modulus - response.relaxed_modulus, atol=1e-10
        )
        assert np.all(response.loss_modulus >= -1e-12)
        if mode == "bulk":
            assert np.max(np.abs(response.loss_modulus)) < 1e-15
        else:
            assert response.affine_modulus > response.relaxed_modulus


def test_frozen_targets_are_immutable():
    mesh = make_regular_hexagonal_mesh(
        5, 4, 0.8, 1.0, gamma_b=-6.0, gamma_c=-8.0, sigma=3.0
    )
    targets = volume_targets(mesh, 0.1)
    original = targets.copy()
    mesh.vertices = mesh.vertices.at[0, 0].add(0.01)
    np.testing.assert_array_equal(targets, original)
    assert not targets.flags.writeable
    assert (
        float(
            mesh.energy_components(target_cell_volumes=targets, volume_penalty=1e4)[
                "volume_penalty"
            ]
        )
        > 0
    )
