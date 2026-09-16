"""Minimal mesh container used by the public Figure 5 notebook."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Self

import jax.numpy as jnp
import numpy as np
from jax import Array

from cvm3d.energy import (
    _cell_parameter,
    _parameters_from_table,
    energy_components as energy_components_from_fields,
    energy_from_tables,
)

__all__ = ["PrismaticMesh3D"]


def _empty_array() -> Array:
    return jnp.array([])


@dataclass
class PrismaticMesh3D:
    """Periodic prismatic mesh and its cell-wise physical parameters."""

    vertices: Array
    half_edges: Array
    faces: Array
    width: float
    height: float
    total_volume: float
    vertices_params: Array = field(default_factory=_empty_array)
    half_edges_params: Array = field(default_factory=_empty_array)
    faces_params: Array = field(default_factory=_empty_array)
    max_edges: int = 20

    @property
    def nb_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def nb_half_edges(self) -> int:
        return int(self.half_edges.shape[0])

    @property
    def nb_edges(self) -> int:
        return self.nb_half_edges // 2

    @property
    def nb_faces(self) -> int:
        return int(self.faces.shape[0])

    @classmethod
    def from_random_seeds(
        cls,
        nb_seeds: int,
        width: float,
        height: float,
        random_key: int,
        total_volume: float | None = None,
        allowed_face_degrees: Iterable[int] | None = None,
        max_attempts: int = 1000,
        rsa_min_distance: float | None = None,
        rsa_max_candidate_attempts: int = 10000,
        lloyd_iterations: int = 0,
        lloyd_relaxation: float = 1.0,
    ) -> Self:
        """Create a periodic Voronoi prism mesh from random RSA seeds."""
        from cvm3d._initializers_core import random_periodic_voronoi_tables

        vertices, half_edges, faces = random_periodic_voronoi_tables(
            nb_seeds,
            width,
            height,
            random_key,
            allowed_face_degrees=allowed_face_degrees,
            max_attempts=max_attempts,
            rsa_min_distance=rsa_min_distance,
            rsa_max_candidate_attempts=rsa_max_candidate_attempts,
            lloyd_iterations=lloyd_iterations,
            lloyd_relaxation=lloyd_relaxation,
        )
        volume = (
            float(total_volume) if total_volume is not None else float(width * height)
        )
        return cls(
            vertices=jnp.asarray(vertices),
            half_edges=jnp.asarray(half_edges, dtype=jnp.int32),
            faces=jnp.asarray(faces, dtype=jnp.int32),
            width=float(width),
            height=float(height),
            total_volume=volume,
        )

    def copy(self) -> Self:
        """Return an independent copy of the mesh arrays."""
        return type(self)(
            vertices=self.vertices.copy(),
            half_edges=self.half_edges.copy(),
            faces=self.faces.copy(),
            width=self.width,
            height=self.height,
            total_volume=self.total_volume,
            vertices_params=self.vertices_params.copy(),
            half_edges_params=self.half_edges_params.copy(),
            faces_params=self.faces_params.copy(),
            max_edges=self.max_edges,
        )

    def set_cell_parameters(
        self,
        gamma_b: Array | float,
        gamma_c: Array | float,
        sigma: Array | float,
        cell_volume: Array | float | None = None,
    ) -> None:
        """Set ``gamma_b``, ``gamma_c``, ``sigma``, and optional ``V0`` columns."""
        columns = [
            _cell_parameter(gamma_b, self.nb_faces, "gamma_b"),
            _cell_parameter(gamma_c, self.nb_faces, "gamma_c"),
            _cell_parameter(sigma, self.nb_faces, "sigma"),
        ]
        if cell_volume is not None:
            cell_volumes = _cell_parameter(cell_volume, self.nb_faces, "cell_volume")
            columns.append(cell_volumes)
            self.total_volume = float(jnp.sum(cell_volumes))
        self.faces_params = jnp.column_stack(columns)
        self.vertices_params = jnp.array([])
        self.half_edges_params = jnp.array([])

    def energy(
        self,
        edge_star_strength: float = 1.0,
        target_cell_volumes: Array | float | None = None,
        volume_penalty: Array | float = 0.0,
    ) -> Array:
        """Evaluate the minimal energy using the mesh parameter tables."""
        return energy_from_tables(
            self.vertices,
            self.half_edges,
            self.faces,
            self.vertices_params,
            self.half_edges_params,
            self.faces_params,
            width=self.width,
            height=self.height,
            total_volume=self.total_volume,
            edge_star_strength=edge_star_strength,
            max_edges=self.max_edges,
            target_cell_volumes=target_cell_volumes,
            volume_penalty=volume_penalty,
        )

    def energy_components(
        self,
        edge_star_strength: float = 1.0,
        target_cell_volumes: Array | float | None = None,
        volume_penalty: Array | float = 0.0,
    ) -> dict[str, Array]:
        """Return the energy components and volume diagnostics."""
        gamma_b, gamma_c, sigma, volumes = _parameters_from_table(
            self.faces_params,
            self.nb_faces,
            self.total_volume,
        )
        return energy_components_from_fields(
            self.vertices,
            self.half_edges,
            self.faces,
            self.width,
            self.height,
            volumes,
            gamma_b,
            gamma_c,
            sigma,
            edge_star_strength=edge_star_strength,
            max_edges=self.max_edges,
            target_cell_volumes=target_cell_volumes,
            volume_penalty=volume_penalty,
        )
