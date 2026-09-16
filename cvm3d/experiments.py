"""Ensemble preparation and response protocol for Figures 5 and S2–S4."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from cvm3d import (
    PrismaticMesh3D,
    complex_modulus,
    make_regular_hexagonal_mesh,
    simulate,
)
from cvm3d.energy import make_energy
from cvm3d.regular import equilibrium_side_length
from cvm3d.relaxation import relax_fixed_topology


@dataclass(frozen=True)
class Parameters:
    gamma_b: float = -6.0
    gamma_c: float = -8.0
    sigma: float = 3.0
    kappa: float = 0.1
    cell_volume: float = 1.0
    mobility: float = 1.0


@dataclass(frozen=True)
class Protocol:
    lloyd_iterations: int = 10
    coarse_iterations: int = 8000
    learning_rate: float = 5e-5
    adam_iterations: int = 50000
    bfgs_max_iterations: int = 10000
    max_retries: int = 10
    newton_iterations: int = 12
    gradient_tolerance: float = 1e-5
    eigenvalue_tolerance: float = 1e-5
    t1_cutoff: float = 0.005


def volume_targets(mesh, kappa):
    targets = np.array(
        mesh.energy_components(edge_star_strength=kappa)["cell_volumes"], copy=True
    )
    if not np.all(np.isfinite(targets) & (targets > 0)):
        raise ValueError("Cell volumes must be positive and finite.")
    np.testing.assert_allclose(targets.sum(), mesh.total_volume, rtol=1e-12, atol=1e-12)
    targets.setflags(write=False)
    return targets


def _energy(mesh, parameters, targets=None, penalty=0.0):
    return make_energy(
        mesh.width,
        mesh.height,
        mesh.total_volume,
        edge_star_strength=parameters.kappa,
        max_edges=mesh.max_edges,
        target_cell_volumes=targets,
        volume_penalty=penalty,
    )


def _polish(mesh, energy, protocol):
    result = relax_fixed_topology(
        mesh,
        energy,
        gradient_tolerance=protocol.gradient_tolerance,
        adam_iterations=protocol.adam_iterations,
        adam_learning_rate=protocol.learning_rate,
        bfgs_max_iterations=protocol.bfgs_max_iterations,
        max_retries=protocol.max_retries,
        newton_iterations=protocol.newton_iterations,
    )
    if not result.converged:
        raise RuntimeError(
            f"Relaxation failed: maximum force = {result.gradient_max_abs:.6g}"
        )
    return result.mesh


def prepare_ensemble(
    nx=5, ny=8, seeds=range(100, 130), *, parameters=Parameters(), protocol=Protocol()
):
    """Relax at K_V=0, then freeze target volumes; retain results in memory."""
    if not jax.config.x64_enabled:
        raise RuntimeError("Enable jax_enable_x64 before constructing meshes.")
    p = parameters
    l0 = equilibrium_side_length(
        gamma_b=p.gamma_b,
        gamma_c=p.gamma_c,
        sigma=p.sigma,
        kappa=p.kappa,
        cell_volume=p.cell_volume,
    )
    regular = make_regular_hexagonal_mesh(
        nx,
        ny,
        l0,
        p.cell_volume,
        gamma_b=p.gamma_b,
        gamma_c=p.gamma_c,
        sigma=p.sigma,
        use_local_cell_volumes=False,
    )
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct realization seeds.")
    meshes, targets = [], []
    for seed in seeds:
        mesh = PrismaticMesh3D.from_random_seeds(
            nb_seeds=nx * ny,
            width=regular.width,
            height=regular.height,
            random_key=seed,
            total_volume=regular.total_volume,
            lloyd_iterations=protocol.lloyd_iterations,
        )
        mesh.set_cell_parameters(gamma_b=p.gamma_b, gamma_c=p.gamma_c, sigma=p.sigma)
        mesh.vertices = jnp.asarray(mesh.vertices, dtype=jnp.float64)
        energy = _energy(mesh, p)
        simulate(
            mesh,
            energy,
            solver=optax.adam(protocol.learning_rate),
            iterations=protocol.coarse_iterations,
            update_t1=True,
            min_dist_t1=protocol.t1_cutoff,
            reject_self_intersections=True,
        )
        mesh = _polish(mesh, energy, protocol)
        meshes.append(mesh)
        targets.append(volume_targets(mesh, p.kappa))
        print(f"Prepared seed {seed}", flush=True)
    return dict(
        regular=regular,
        meshes=meshes,
        targets=targets,
        seeds=seeds,
        parameters=p,
        protocol=protocol,
        l0=l0,
    )


def measure_ensemble(prepared, volume_penalty=1e4, angular_frequencies=None):
    """Polish copies at fixed targets and measure bulk and engineering shear."""
    if not np.isfinite(volume_penalty) or volume_penalty < 0:
        raise ValueError("volume_penalty must be finite and non-negative.")
    p, protocol = prepared["parameters"], prepared["protocol"]
    frequencies = (
        np.logspace(-3, 10, 111)
        if angular_frequencies is None
        else np.asarray(angular_frequencies)
    )
    options = dict(
        edge_star_strength=p.kappa,
        volume_penalty=volume_penalty,
        mobility=p.mobility,
        eigenvalue_tolerance=protocol.eigenvalue_tolerance,
        gradient_tolerance=protocol.gradient_tolerance,
        require_equilibrium=True,
    )

    def responses(mesh, targets):
        return [
            complex_modulus(
                mesh, frequencies, mode=mode, target_cell_volumes=targets, **options
            )
            for mode in ("bulk", "simple_shear")
        ]

    regular = prepared["regular"]
    rb, rs = responses(regular, volume_targets(regular, p.kappa))
    positive = rs.hessian_eigenvalues[
        rs.hessian_eigenvalues > protocol.eigenvalue_tolerance
    ]
    h0 = float(np.median(positive))
    if not np.isfinite(h0) or h0 <= 0:
        raise RuntimeError("No positive regular-reference Hessian scale.")
    meshes, bulk, shear = [], [], []
    for seed, mesh, targets in zip(
        prepared["seeds"], prepared["meshes"], prepared["targets"], strict=True
    ):
        polished = _polish(mesh, _energy(mesh, p, targets, volume_penalty), protocol)
        b, s = responses(polished, targets)
        meshes.append(polished)
        bulk.append(b)
        shear.append(s)
        print(f"Measured seed {seed}, K_V={volume_penalty:g}", flush=True)
    return dict(
        **{k: v for k, v in prepared.items() if k != "meshes"},
        meshes=meshes,
        bulk=bulk,
        shear=shear,
        regular_bulk=rb,
        regular_shear=rs,
        h0=h0,
        omega0=p.mobility * h0,
        frequencies=frequencies,
        volume_penalty=volume_penalty,
    )
