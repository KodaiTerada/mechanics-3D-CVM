"""JAX geometry for prismatic 3D CVMs on periodic 2D vertex meshes."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["face_vertices_unfolded"]
from jax import Array

EPSILON = 1e-12


@jax.jit
def edge_vector(
    half_edge_id: Array,
    vertices: Array,
    half_edges: Array,
    width: float,
    height: float,
    box_matrix: Array | None = None,
) -> Array:
    """Return the periodic in-plane vector of a half-edge."""
    source = half_edges.at[half_edge_id, 3].get()
    target = half_edges.at[half_edge_id, 4].get()
    offset = (
        half_edges.at[half_edge_id, 6:8].get()
        @ (jnp.diag(jnp.array([width, height])) if box_matrix is None else box_matrix).T
    )
    return vertices.at[target, :2].get() + offset - vertices.at[source, :2].get()


@jax.jit
def edge_length(
    half_edge_id: Array,
    vertices: Array,
    half_edges: Array,
    width: float,
    height: float,
    box_matrix: Array | None = None,
) -> Array:
    """Return the periodic in-plane length of a half-edge."""
    vector = edge_vector(
        half_edge_id, vertices, half_edges, width, height, box_matrix=box_matrix
    )
    return jnp.linalg.norm(vector)


@partial(jax.jit, static_argnames=("max_edges",))
def face_vertices_unfolded(
    face_id: Array,
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
    box_matrix: Array | None = None,
) -> tuple[Array, Array]:
    """Return unfolded 2D vertices and a validity mask.

    Optional box_matrix columns are lattice vectors in Cartesian coordinates;
    integer half-edge image offsets are preserved under shear."""
    start_he = faces.at[face_id].get()
    box = jnp.diag(jnp.array([width, height])) if box_matrix is None else box_matrix

    def scan_body(
        carry: tuple[Array, Array, Array], _: Array
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        current_he, cumulative_offset, stopped = carry
        source = half_edges.at[current_he, 3].get()
        point = vertices.at[source, :2].get() + cumulative_offset
        valid = jnp.logical_not(stopped)

        next_he = half_edges.at[current_he, 1].get()
        he_offset = half_edges.at[current_he, 6:8].get() @ box.T
        next_offset = cumulative_offset + he_offset
        next_stopped = stopped | (next_he == start_he)
        return (next_he, next_offset, next_stopped), (point, valid)

    initial_carry = (start_he, jnp.zeros(2), jnp.array(False))
    _, (points, valid) = jax.lax.scan(
        scan_body, initial_carry, xs=jnp.arange(max_edges)
    )
    return points, valid


@jax.jit
def _cross2d(a: Array, b: Array) -> Array:
    return a[0] * b[1] - a[1] * b[0]


@partial(jax.jit, static_argnames=("max_edges",))
def face_self_intersection_penalty(
    face_id: Array,
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
    softness: float = 1e-4,
) -> Array:
    """Return a soft penalty for self-intersections inside one polygonal face."""
    points, valid_vertices = face_vertices_unfolded(
        face_id,
        vertices,
        half_edges,
        faces,
        width,
        height,
        max_edges=max_edges,
    )
    ids = jnp.arange(max_edges)
    count = jnp.sum(valid_vertices.astype(jnp.int32))
    next_points = jnp.roll(points, -1, axis=0)
    segment_starts = points
    segment_ends = jnp.where((ids == count - 1)[:, None], points[0], next_points)
    segment_valid = valid_vertices & (ids < count)

    i = ids[:, None]
    j = ids[None, :]
    valid_pair = segment_valid[:, None] & segment_valid[None, :]
    valid_pair = valid_pair & (i < j)
    valid_pair = valid_pair & (j != i + 1)
    valid_pair = valid_pair & ~((i == 0) & (j == count - 1))

    def pair_penalty(pair: Array) -> Array:
        edge_i, edge_j = pair
        a = segment_starts[edge_i]
        b = segment_ends[edge_i]
        c = segment_starts[edge_j]
        d = segment_ends[edge_j]
        o1 = _cross2d(b - a, c - a)
        o2 = _cross2d(b - a, d - a)
        o3 = _cross2d(d - c, a - c)
        o4 = _cross2d(d - c, b - c)
        scaled_softness = jnp.asarray(softness, dtype=points.dtype)
        side_a = jax.nn.softplus(-(o1 * o2) / scaled_softness) * scaled_softness
        side_b = jax.nn.softplus(-(o3 * o4) / scaled_softness) * scaled_softness
        return side_a * side_b / (scaled_softness + EPSILON)

    pairs = jnp.stack(
        [
            jnp.broadcast_to(i, (max_edges, max_edges)),
            jnp.broadcast_to(j, (max_edges, max_edges)),
        ],
        axis=-1,
    )
    penalties = jax.vmap(jax.vmap(pair_penalty))(pairs)
    return jnp.sum(jnp.where(valid_pair, penalties, 0.0))


@partial(jax.jit, static_argnames=("max_edges",))
def face_has_self_intersection(
    face_id: Array,
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
) -> Array:
    """Return True if one face has a strict crossing between non-adjacent edges."""
    points, valid_vertices = face_vertices_unfolded(
        face_id,
        vertices,
        half_edges,
        faces,
        width,
        height,
        max_edges=max_edges,
    )
    ids = jnp.arange(max_edges)
    count = jnp.sum(valid_vertices.astype(jnp.int32))
    next_points = jnp.roll(points, -1, axis=0)
    segment_starts = points
    segment_ends = jnp.where((ids == count - 1)[:, None], points[0], next_points)
    segment_valid = valid_vertices & (ids < count)

    i = ids[:, None]
    j = ids[None, :]
    valid_pair = segment_valid[:, None] & segment_valid[None, :]
    valid_pair = valid_pair & (i < j)
    valid_pair = valid_pair & (j != i + 1)
    valid_pair = valid_pair & ~((i == 0) & (j == count - 1))

    def pair_crosses(pair: Array) -> Array:
        edge_i, edge_j = pair
        a = segment_starts[edge_i]
        b = segment_ends[edge_i]
        c = segment_starts[edge_j]
        d = segment_ends[edge_j]
        o1 = _cross2d(b - a, c - a)
        o2 = _cross2d(b - a, d - a)
        o3 = _cross2d(d - c, a - c)
        o4 = _cross2d(d - c, b - c)
        return (o1 * o2 < 0.0) & (o3 * o4 < 0.0)

    pairs = jnp.stack(
        [
            jnp.broadcast_to(i, (max_edges, max_edges)),
            jnp.broadcast_to(j, (max_edges, max_edges)),
        ],
        axis=-1,
    )
    crossings = jax.vmap(jax.vmap(pair_crosses))(pairs)
    return jnp.any(valid_pair & crossings)


@partial(jax.jit, static_argnames=("max_edges",))
def has_self_intersections(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
) -> Array:
    """Return True if any cell polygon has a strict self-intersection."""
    return jnp.any(
        jax.vmap(
            lambda face_id: face_has_self_intersection(
                face_id,
                vertices,
                half_edges,
                faces,
                width,
                height,
                max_edges=max_edges,
            )
        )(jnp.arange(faces.shape[0]))
    )


@partial(jax.jit, static_argnames=("max_edges",))
def cell_self_intersection_penalty(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
    softness: float = 1e-4,
) -> Array:
    """Return summed soft self-intersection penalty over all cells."""
    return jnp.sum(
        jax.vmap(
            lambda face_id: face_self_intersection_penalty(
                face_id,
                vertices,
                half_edges,
                faces,
                width,
                height,
                max_edges=max_edges,
                softness=softness,
            )
        )(jnp.arange(faces.shape[0]))
    )


@partial(jax.jit, static_argnames=("max_edges",))
def face_area(
    face_id: Array,
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
    box_matrix: Array | None = None,
) -> Array:
    """Return the area of a periodic polygonal cell using the shoelace formula."""
    start_he = faces.at[face_id].get()

    def contribution(
        half_edge_id: Array, cumulative_offset: Array
    ) -> tuple[Array, Array]:
        source = half_edges.at[half_edge_id, 3].get()
        target = half_edges.at[half_edge_id, 4].get()
        he_offset = (
            half_edges.at[half_edge_id, 6:8].get()
            @ (
                jnp.diag(jnp.array([width, height]))
                if box_matrix is None
                else box_matrix
            ).T
        )
        xy0 = vertices.at[source, :2].get() + cumulative_offset
        xy1 = vertices.at[target, :2].get() + cumulative_offset + he_offset
        cross = xy0[0] * xy1[1] - xy1[0] * xy0[1]
        return cross, he_offset

    cross0, offset0 = contribution(start_he, jnp.zeros(2))
    initial_carry = (start_he, cross0, offset0, False)

    def scan_body(
        carry: tuple[Array, Array, Array, Array], _: Array
    ) -> tuple[tuple[Array, Array, Array, Array], None]:
        previous_he, previous_cross, previous_offset, stopped = carry
        current_he = half_edges.at[previous_he, 1].get()
        is_start = current_he == start_he
        stopped = stopped | is_start
        cross, he_offset = contribution(current_he, previous_offset)
        next_cross = jax.lax.select(stopped, previous_cross, previous_cross + cross)
        next_offset = jax.lax.select(
            stopped, previous_offset, previous_offset + he_offset
        )
        return (current_he, next_cross, next_offset, stopped), None

    (_, cross_sum, _, _), _ = jax.lax.scan(
        scan_body, initial_carry, xs=jnp.arange(max_edges - 1)
    )
    return 0.5 * jnp.abs(cross_sum)


@partial(jax.jit, static_argnames=("max_edges",))
def face_perimeter(
    face_id: Array,
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
    box_matrix: Array | None = None,
) -> Array:
    """Return the perimeter of a periodic polygonal cell."""
    start_he = faces.at[face_id].get()
    length0 = edge_length(
        start_he, vertices, half_edges, width, height, box_matrix=box_matrix
    )
    initial_carry = (start_he, length0, False)

    def scan_body(
        carry: tuple[Array, Array, Array], _: Array
    ) -> tuple[tuple[Array, Array, Array], None]:
        previous_he, previous_length, stopped = carry
        current_he = half_edges.at[previous_he, 1].get()
        is_start = current_he == start_he
        stopped = stopped | is_start
        current_length = edge_length(
            current_he, vertices, half_edges, width, height, box_matrix=box_matrix
        )
        next_length = jax.lax.select(
            stopped, previous_length, previous_length + current_length
        )
        return (current_he, next_length, stopped), None

    (_, perimeter, _), _ = jax.lax.scan(
        scan_body, initial_carry, xs=jnp.arange(max_edges - 1)
    )
    return perimeter


@partial(jax.jit, static_argnames=("max_edges",))
def cell_areas(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
    box_matrix: Array | None = None,
) -> Array:
    """Return all cell basal areas."""

    def mapped(face_id: Array) -> Array:
        return face_area(
            face_id,
            vertices,
            half_edges,
            faces,
            width,
            height,
            max_edges=max_edges,
            box_matrix=box_matrix,
        )

    return jax.vmap(mapped)(jnp.arange(faces.shape[0]))


@partial(jax.jit, static_argnames=("max_edges",))
def cell_perimeters(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
    box_matrix: Array | None = None,
) -> Array:
    """Return all cell apical/basal perimeters."""

    def mapped(face_id: Array) -> Array:
        return face_perimeter(
            face_id,
            vertices,
            half_edges,
            faces,
            width,
            height,
            max_edges=max_edges,
            box_matrix=box_matrix,
        )

    return jax.vmap(mapped)(jnp.arange(faces.shape[0]))


@partial(jax.jit, static_argnames=("max_edges",))
def face_shape_tensor(
    face_id: Array,
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
) -> Array:
    """Return the in-plane vertex second-moment tensor of one periodic cell."""
    start_he = faces.at[face_id].get()
    box = jnp.array([width, height])

    def scan_body(
        carry: tuple[Array, Array, Array], _: Array
    ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array]]:
        current_he, cumulative_offset, stopped = carry
        source = half_edges.at[current_he, 3].get()
        point = vertices.at[source, :2].get() + cumulative_offset
        valid = jnp.logical_not(stopped)

        next_he = half_edges.at[current_he, 1].get()
        he_offset = half_edges.at[current_he, 6:8].get() * box
        next_offset = cumulative_offset + he_offset
        next_stopped = stopped | (next_he == start_he)
        return (next_he, next_offset, next_stopped), (point, valid)

    initial_carry = (start_he, jnp.zeros(2), jnp.array(False))
    _, (points, valid) = jax.lax.scan(
        scan_body, initial_carry, xs=jnp.arange(max_edges)
    )
    weights = valid.astype(points.dtype)
    count = jnp.sum(weights) + EPSILON
    centroid = jnp.sum(points * weights[:, None], axis=0) / count
    centered = (points - centroid) * weights[:, None]
    return centered.T @ centered / count


@partial(jax.jit, static_argnames=("max_edges",))
def cell_shape_tensors(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
) -> Array:
    """Return in-plane vertex second-moment tensors for all cells."""

    def mapped(face_id: Array) -> Array:
        return face_shape_tensor(
            face_id, vertices, half_edges, faces, width, height, max_edges=max_edges
        )

    return jax.vmap(mapped)(jnp.arange(faces.shape[0]))


@jax.jit
def half_edge_lengths(
    vertices: Array, half_edges: Array, width: float, height: float
) -> Array:
    """Return lengths for all half-edges."""

    def mapped(half_edge_id: Array) -> Array:
        return edge_length(half_edge_id, vertices, half_edges, width, height)

    return jax.vmap(mapped)(jnp.arange(half_edges.shape[0]))


@partial(jax.jit, static_argnames=("max_edges",))
def total_basal_area(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    max_edges: int = 20,
) -> Array:
    """Return total basal area of the periodic monolayer."""
    return jnp.sum(
        cell_areas(vertices, half_edges, faces, width, height, max_edges=max_edges)
    )


@jax.jit
def unique_junction_length(
    vertices: Array, half_edges: Array, width: float, height: float
) -> Array:
    """Return total unique apical junction length.

    The DCEL stores two half-edges per interface. Summing all half-edge lengths and
    multiplying by 1/2 counts each physical junction once.
    """
    return 0.5 * jnp.sum(half_edge_lengths(vertices, half_edges, width, height))


@jax.jit
def global_height(total_volume: Array, basal_area: Array) -> Array:
    """Return the volume-conserving monolayer height."""
    return total_volume / (basal_area + EPSILON)


@jax.jit
def cell_heights(face_areas: Array, cell_volumes: Array) -> Array:
    """Return per-cell prism heights from local volume conservation."""
    return cell_volumes / (face_areas + EPSILON)


@jax.jit
def contact_area(
    vertices: Array,
    half_edges: Array,
    width: float,
    height: float,
    monolayer_height: Array,
) -> Array:
    """Return total cell-cell contact area for vertical prismatic interfaces."""
    return (
        unique_junction_length(vertices, half_edges, width, height) * monolayer_height
    )


@jax.jit
def update_pbc(
    vertices: Array, half_edges: Array, faces: Array, width: float, height: float
) -> tuple[Array, Array, Array]:
    """Wrap vertices back into the periodic box and update half-edge offsets."""

    def half_edge_offsets(half_edge_id: Array) -> tuple[Array, Array, Array, Array]:
        target = half_edges.at[half_edge_id, 4].get()
        target_xy = vertices.at[target, :2].get()
        target_x = jnp.where(
            target_xy[0] < 0.0, -1, jnp.where(target_xy[0] > width, 1, 0)
        )
        target_y = jnp.where(
            target_xy[1] < 0.0, -1, jnp.where(target_xy[1] > height, 1, 0)
        )

        source = half_edges.at[half_edge_id, 3].get()
        source_xy = vertices.at[source, :2].get()
        source_x = jnp.where(
            source_xy[0] < 0.0, -1, jnp.where(source_xy[0] > width, 1, 0)
        )
        source_y = jnp.where(
            source_xy[1] < 0.0, -1, jnp.where(source_xy[1] > height, 1, 0)
        )
        return target_x, target_y, source_x, source_y

    target_x, target_y, source_x, source_y = jax.vmap(half_edge_offsets)(
        jnp.arange(half_edges.shape[0])
    )
    half_edges = half_edges.at[:, 6].add(target_x - source_x)
    half_edges = half_edges.at[:, 7].add(target_y - source_y)

    x = vertices[:, 0]
    y = vertices[:, 1]
    x = jnp.where(x < 0.0, x + width, jnp.where(x > width, x - width, x))
    y = jnp.where(y < 0.0, y + height, jnp.where(y > height, y - height, y))
    vertices = vertices.at[:, 0].set(x)
    vertices = vertices.at[:, 1].set(y)
    return vertices, half_edges, faces


def affine_scale_vertices(vertices: Array, scale_x: float, scale_y: float) -> Array:
    """Return in-plane vertices after an affine box-scaled deformation."""
    return vertices * jnp.array([scale_x, scale_y])
