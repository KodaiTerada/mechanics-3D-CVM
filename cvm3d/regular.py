"""Regular-cell energy and scalar equilibrium for the corner model."""

import numpy as np
from scipy.optimize import minimize_scalar


def regular_cell_energy(l, *, gamma_b, gamma_c, sigma, kappa, cell_volume=1.0):
    l = np.asarray(l)
    return (
        1.5 * np.sqrt(3) * gamma_b * l**2
        + 2 * gamma_c * cell_volume / (np.sqrt(3) * l)
        + 6 * sigma * l
        + kappa * (32 / (3 * l**2) + 27 * l**4 / cell_volume**2)
    )


def equilibrium_side_length(*, bounds=(1e-4, 1e2), samples=1000, **parameters):
    """Lowest-energy scalar minimum found in bounds; includes competing wells.

    This minimizes within regular tilings, not over all vertex configurations.
    """
    grid = np.linspace(np.log(bounds[0]), np.log(bounds[1]), samples)
    energies = regular_cell_energy(np.exp(grid), **parameters)
    if np.argmin(energies) in (0, samples - 1):
        raise ValueError("Increase side-length search bounds")
    candidates = (
        np.flatnonzero(
            (energies[1:-1] < energies[:-2]) & (energies[1:-1] < energies[2:])
        )
        + 1
    )
    results = [
        minimize_scalar(
            lambda x: regular_cell_energy(np.exp(x), **parameters),
            bounds=(grid[i - 1], grid[i + 1]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        for i in candidates
    ]
    if not results or not all(r.success for r in results):
        raise RuntimeError("Regular equilibrium search failed")
    return float(np.exp(min(results, key=lambda r: r.fun).x))
