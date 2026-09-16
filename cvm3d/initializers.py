"""Initializers returning :mod:`cvm3d` meshes."""

from __future__ import annotations

from cvm3d._initializers_core import (
    make_regular_hexagonal_mesh as _make_regular_hexagonal_mesh,
)

from cvm3d.mesh import PrismaticMesh3D


def make_regular_hexagonal_mesh(
    nx: int,
    ny: int,
    side_length: float,
    cell_volume: float,
    *,
    gamma_b: float,
    gamma_c: float,
    sigma: float,
    use_local_cell_volumes: bool = False,
    max_edges: int = 20,
) -> PrismaticMesh3D:
    """Return a regular mesh with the required minimal parameter table."""
    return _make_regular_hexagonal_mesh(
        nx,
        ny,
        side_length,
        cell_volume,
        gamma_b=gamma_b,
        gamma_c=gamma_c,
        sigma=sigma,
        use_local_cell_volumes=use_local_cell_volumes,
        max_edges=max_edges,
    )


__all__ = ["make_regular_hexagonal_mesh"]
