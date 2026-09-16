"""Minimal edge-star energy for periodic prismatic cell meshes."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from cvm3d.geometry import (
    cell_areas,
    cell_heights,
    cell_perimeters,
    face_vertices_unfolded,
    global_height,
)

InnerEnergy = Callable[[Array, Array, Array, Array, Array, Array], Array]

# Per-cell-corner model: (kappa / 3) Tr[(sum_alpha d_alpha d_alpha^T)^-1].
EDGE_STAR_MODEL = "cell_corner_inverse_sum_over_3_v1"

__all__ = ["make_energy"]


def _cell_parameter(value: Array | float, face_count: int, name: str) -> Array:
    array = jnp.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return jnp.full((face_count,), jnp.ravel(array)[0])
    if array.ndim == 1 and array.shape[0] == face_count:
        return array
    raise ValueError(f"{name} must be scalar or have shape (nb_faces,).")


def _cell_heights(face_areas: Array, total_volume: Array | float) -> Array:
    volumes = jnp.asarray(total_volume)
    if volumes.ndim == 0 or volumes.size == 1:
        height = global_height(jnp.ravel(volumes)[0], jnp.sum(face_areas))
        return jnp.full((face_areas.shape[0],), height)
    if volumes.ndim == 1 and volumes.shape[0] == face_areas.shape[0]:
        return cell_heights(face_areas, volumes)
    raise ValueError("total_volume must be scalar or have shape (nb_faces,).")


def _cell_volume_penalty_components(
    face_areas: Array,
    cell_height_values: Array,
    target_cell_volumes: Array | float | None,
    volume_penalty: Array | float,
) -> tuple[Array, Array, Array]:
    """Return cell volumes, relative target errors, and their soft penalty.

    ``cell_height_values`` may be either a common scalar height or one height
    per cell.  Omitting ``target_cell_volumes`` disables the constraint and
    returns zero relative errors, preserving the pre-penalty energy exactly.
    """
    cell_volumes = face_areas * jnp.asarray(cell_height_values)
    if target_cell_volumes is None:
        relative_errors = jnp.zeros_like(face_areas)
        penalty = jnp.asarray(0.0, dtype=face_areas.dtype)
    else:
        targets = _cell_parameter(
            target_cell_volumes,
            face_areas.shape[0],
            "target_cell_volumes",
        )
        relative_errors = cell_volumes / targets - 1.0
        penalty = 0.5 * jnp.asarray(volume_penalty) * jnp.sum(relative_errors**2)
    return cell_volumes, relative_errors, penalty


def _edge_star_energy(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    cell_height_values: Array,
    max_edges: int,
    box_matrix: Array | None = None,
) -> Array:
    """Return sum_corner Tr[(d1 d1^T + d2 d2^T + d3 d3^T)^-1]/3.

    Each cell corner is counted on both layers, including shared corners.
    For vertical prisms d3=(0,0,h). No eigenvalue clipping is applied.
    """

    def face_energy(face_id: Array, cell_height: Array) -> Array:
        points, valid = face_vertices_unfolded(
            face_id,
            vertices,
            half_edges,
            faces,
            width,
            height,
            max_edges=max_edges,
            box_matrix=box_matrix,
        )
        ids = jnp.arange(max_edges)
        count = jnp.sum(valid.astype(jnp.int32))
        previous_ids = jnp.where(ids == 0, count - 1, ids - 1)
        next_ids = jnp.where(ids == count - 1, 0, ids + 1)
        previous_vectors = points[previous_ids] - points
        next_vectors = points[next_ids] - points
        in_plane_stars = jnp.einsum(
            "ni,nj->nij", previous_vectors, previous_vectors
        ) + jnp.einsum("ni,nj->nij", next_vectors, next_vectors)
        identity = jnp.eye(2, dtype=vertices.dtype)
        safe_stars = jnp.where(valid[:, None, None], in_plane_stars, identity)
        lateral = jnp.trace(jnp.linalg.inv(safe_stars), axis1=-2, axis2=-1)
        vertical = 1.0 / cell_height**2
        weighted_trace = (lateral + vertical) / 3.0
        return 2.0 * jnp.sum(jnp.where(valid, weighted_trace, 0.0))

    return jnp.sum(
        jax.vmap(face_energy)(jnp.arange(faces.shape[0]), cell_height_values)
    )


def energy_prismatic(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    total_volume: Array | float,
    gamma_b: Array | float,
    gamma_c: Array | float,
    sigma: Array | float,
    edge_star_strength: float = 1.0,
    max_edges: int = 20,
    target_cell_volumes: Array | float | None = None,
    volume_penalty: Array | float = 0.0,
    box_matrix: Array | None = None,
) -> Array:
    """Return ``basal + contact + line + edge-star`` energy.

    ``gamma_b``, ``gamma_c``, and ``sigma`` may be scalar or cell-wise.
    ``total_volume`` may be a scalar global volume or a cell-wise volume
    vector. When ``target_cell_volumes`` is supplied, the additional energy is
    ``volume_penalty / 2 * sum((A_i h_i / V0_i - 1)**2)``. In particular,
    scalar ``total_volume`` together with cell-wise targets implements common
    height plus soft local volume conservation. Other soft geometry penalties
    are not included.
    """
    areas = cell_areas(
        vertices,
        half_edges,
        faces,
        width,
        height,
        max_edges=max_edges,
        box_matrix=box_matrix,
    )
    perimeters = cell_perimeters(
        vertices,
        half_edges,
        faces,
        width,
        height,
        max_edges=max_edges,
        box_matrix=box_matrix,
    )
    heights = _cell_heights(areas, total_volume)
    gamma_b_cells = _cell_parameter(gamma_b, faces.shape[0], "gamma_b")
    gamma_c_cells = _cell_parameter(gamma_c, faces.shape[0], "gamma_c")
    sigma_cells = _cell_parameter(sigma, faces.shape[0], "sigma")

    basal = jnp.sum(gamma_b_cells * areas)
    contact = 0.5 * jnp.sum(gamma_c_cells * heights * perimeters)
    # ``sigma`` is the cell-autonomous apical constriction coefficient.  It is
    # therefore multiplied by the complete perimeter of every cell.  This is
    # intentionally different from an interfacial line tension, for which the
    # two directed copies of a shared edge would require a factor of one half.
    line = jnp.sum(sigma_cells * perimeters)
    edge_star = edge_star_strength * _edge_star_energy(
        vertices,
        half_edges,
        faces,
        width,
        height,
        heights,
        max_edges,
        box_matrix=box_matrix,
    )
    _, _, volume_penalty_energy = _cell_volume_penalty_components(
        areas,
        heights,
        target_cell_volumes,
        volume_penalty,
    )
    return basal + contact + line + edge_star + volume_penalty_energy


def _parameters_from_table(
    face_params: Array, face_count: int, total_volume: Array | float
) -> tuple[Array, Array, Array, Array | float]:
    params = jnp.asarray(face_params)
    if (
        params.ndim != 2
        or params.shape[0] != face_count
        or params.shape[1] not in (3, 4)
    ):
        raise ValueError("face_params must have shape (nb_faces, 3) or (nb_faces, 4).")
    volumes = params[:, 3] if params.shape[1] == 4 else total_volume
    return params[:, 0], params[:, 1], params[:, 2], volumes


def energy_from_tables(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    vertex_params: Array,
    half_edge_params: Array,
    face_params: Array,
    *,
    width: float,
    height: float,
    total_volume: Array | float,
    edge_star_strength: float = 1.0,
    max_edges: int = 20,
    target_cell_volumes: Array | float | None = None,
    volume_penalty: Array | float = 0.0,
    box_matrix: Array | None = None,
) -> Array:
    """Evaluate the minimal energy from three or four face columns.

    The columns are ``gamma_b``, ``gamma_c``, ``sigma``, and optionally the
    cell-wise volume ``V0``. Vertex and half-edge parameter tables are ignored;
    they remain in the signature only so the function can be passed to the
    shared simulator.
    """
    del vertex_params, half_edge_params
    gamma_b, gamma_c, sigma, volumes = _parameters_from_table(
        face_params, faces.shape[0], total_volume
    )
    return energy_prismatic(
        vertices,
        half_edges,
        faces,
        width,
        height,
        volumes,
        gamma_b,
        gamma_c,
        sigma,
        edge_star_strength=edge_star_strength,
        max_edges=max_edges,
        box_matrix=box_matrix,
        target_cell_volumes=target_cell_volumes,
        volume_penalty=volume_penalty,
    )


def make_energy(
    width: float,
    height: float,
    total_volume: Array | float,
    edge_star_strength: float = 1.0,
    max_edges: int = 20,
    target_cell_volumes: Array | float | None = None,
    volume_penalty: Array | float = 0.0,
) -> InnerEnergy:
    """Create the table-based energy expected by :func:`simulate`.

    ``target_cell_volumes`` and ``volume_penalty`` are captured by the closure,
    so target volumes need not be stored in the face table. This keeps the
    fourth face column's existing local-height meaning unchanged.
    """

    def energy(
        vertices: Array,
        half_edges: Array,
        faces: Array,
        vertex_params: Array,
        half_edge_params: Array,
        face_params: Array,
    ) -> Array:
        return energy_from_tables(
            vertices,
            half_edges,
            faces,
            vertex_params,
            half_edge_params,
            face_params,
            width=width,
            height=height,
            total_volume=total_volume,
            edge_star_strength=edge_star_strength,
            max_edges=max_edges,
            target_cell_volumes=target_cell_volumes,
            volume_penalty=volume_penalty,
        )

    return energy


def energy_components(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    total_volume: Array | float,
    gamma_b: Array | float,
    gamma_c: Array | float,
    sigma: Array | float,
    edge_star_strength: float = 1.0,
    max_edges: int = 20,
    target_cell_volumes: Array | float | None = None,
    volume_penalty: Array | float = 0.0,
) -> dict[str, Array]:
    """Return the energy components and basic geometry."""
    areas = cell_areas(vertices, half_edges, faces, width, height, max_edges=max_edges)
    perimeters = cell_perimeters(
        vertices, half_edges, faces, width, height, max_edges=max_edges
    )
    heights = _cell_heights(areas, total_volume)
    gamma_b_cells = _cell_parameter(gamma_b, faces.shape[0], "gamma_b")
    gamma_c_cells = _cell_parameter(gamma_c, faces.shape[0], "gamma_c")
    sigma_cells = _cell_parameter(sigma, faces.shape[0], "sigma")
    basal = jnp.sum(gamma_b_cells * areas)
    contact = 0.5 * jnp.sum(gamma_c_cells * heights * perimeters)
    line = jnp.sum(sigma_cells * perimeters)
    edge_star = edge_star_strength * _edge_star_energy(
        vertices,
        half_edges,
        faces,
        width,
        height,
        heights,
        max_edges,
    )
    cell_volumes, relative_volume_errors, volume_penalty_energy = (
        _cell_volume_penalty_components(
            areas,
            heights,
            target_cell_volumes,
            volume_penalty,
        )
    )
    return {
        "basal": basal,
        "contact": contact,
        "line": line,
        "edge_star": edge_star,
        "volume_penalty": volume_penalty_energy,
        "total": basal + contact + line + edge_star + volume_penalty_energy,
        "basal_area": jnp.sum(areas),
        "height": heights,
        "cell_volumes": cell_volumes,
        "relative_volume_errors": relative_volume_errors,
    }
