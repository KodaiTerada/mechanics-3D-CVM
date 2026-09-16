"""Internal topology updates for periodic prismatic CVM meshes."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from cvm3d.geometry import EPSILON, edge_length, update_pbc

InnerEnergy = Callable[[Array, Array, Array, Array, Array, Array], Array]


def do_not_update_t1(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    vertex_params: Array,
    half_edge_params: Array,
    face_params: Array,
    energy_fn: InnerEnergy,
    min_dist_t1: float,
    selected_vertices: Array,
    selected_half_edges: Array,
    selected_faces: Array,
    t1_invalid_degree_penalty: float,
    min_face_vertices_after_t1: int,
) -> tuple[Array, Array, Array]:
    """T1 no-op with the same signature as ``update_t1``."""
    del (
        width,
        height,
        vertex_params,
        half_edge_params,
        face_params,
        energy_fn,
        min_dist_t1,
    )
    del selected_vertices, selected_half_edges, selected_faces
    del t1_invalid_degree_penalty, min_face_vertices_after_t1
    return vertices, half_edges, faces


def _rescaled_source_position(
    vertices: Array,
    half_edges: Array,
    half_edge_id: Array,
    width: float,
    height: float,
    min_dist_t1: float,
    angle: Array,
) -> Array:
    """Return a source endpoint after rotating and rescaling a short edge."""
    half_edge = half_edges.at[half_edge_id].get()
    source = half_edge.at[3].get()
    target = half_edge.at[4].get()
    box = jnp.array([width, height])
    source_xy = vertices.at[source, :2].get()
    target_xy = (
        vertices.at[target, :2].get() + half_edges.at[half_edge_id, 6:8].get() * box
    )
    center = 0.5 * (source_xy + target_xy)
    relative = source_xy - center
    rotated = jnp.array(
        [
            jnp.cos(angle) * relative[0] - jnp.sin(angle) * relative[1],
            jnp.sin(angle) * relative[0] + jnp.cos(angle) * relative[1],
        ]
    )
    norm = jnp.linalg.norm(rotated)
    direction = jnp.where(
        norm > EPSILON, rotated / (norm + EPSILON), jnp.array([1.0, 0.0])
    )
    return center + direction * (1.1 * min_dist_t1 / 2.0)


def _set_t1_vertex_positions(
    vertices: Array,
    half_edges: Array,
    he_idx: Array,
    twin_he_idx: Array,
    width: float,
    height: float,
    min_dist_t1: float,
    angle: Array,
) -> Array:
    """Place the two endpoints of a candidate short edge state."""
    he = half_edges.at[he_idx].get()
    twin_he = half_edges.at[twin_he_idx].get()
    source_position = _rescaled_source_position(
        vertices, half_edges, he_idx, width, height, min_dist_t1, angle
    )
    twin_source_position = _rescaled_source_position(
        vertices, half_edges, twin_he_idx, width, height, min_dist_t1, angle
    )
    vertices = vertices.at[he[3], :2].set(source_position)
    vertices = vertices.at[he[4], :2].set(twin_source_position)
    vertices = vertices.at[twin_he[3], :2].set(twin_source_position)
    vertices = vertices.at[twin_he[4], :2].set(source_position)
    return vertices


@partial(jax.jit, static_argnames=("max_edges",))
def _faces_close_within_max_edges(
    half_edges: Array, faces: Array, max_edges: int
) -> Array:
    """Return True when every face cycle closes within ``max_edges`` next-links."""

    def face_closes(start_he: Array) -> Array:
        def scan_body(
            carry: tuple[Array, Array], _: Array
        ) -> tuple[tuple[Array, Array], None]:
            current_he, closed = carry
            next_he = half_edges.at[current_he, 1].get()
            return (next_he, closed | (next_he == start_he)), None

        (_, closed), _ = jax.lax.scan(
            scan_body, (start_he, jnp.array(False)), xs=jnp.arange(max_edges)
        )
        return closed

    return jnp.all(jax.vmap(face_closes)(faces))


@partial(jax.jit, static_argnames=("max_edges", "min_vertices"))
def _faces_have_minimum_vertices(
    half_edges: Array, faces: Array, max_edges: int, min_vertices: int
) -> Array:
    """Return True when every face closes and has at least ``min_vertices`` sides."""

    def face_is_valid(start_he: Array) -> Array:
        def scan_body(
            carry: tuple[Array, Array, Array], _: Array
        ) -> tuple[tuple[Array, Array, Array], None]:
            current_he, closed, count = carry
            next_he = half_edges.at[current_he, 1].get()
            step_valid = jnp.logical_not(closed)
            next_count = count + step_valid.astype(jnp.int32)
            next_closed = closed | (next_he == start_he)
            return (next_he, next_closed, next_count), None

        (_, closed, count), _ = jax.lax.scan(
            scan_body,
            (start_he, jnp.array(False), jnp.array(0, dtype=jnp.int32)),
            xs=jnp.arange(max_edges),
        )
        return closed & (count >= min_vertices)

    return jnp.all(jax.vmap(face_is_valid)(faces))


@jax.jit
def _half_edge_links_are_consistent(half_edges: Array) -> Array:
    """Return True when previous/next links are mutual inverses."""
    half_edge_ids = jnp.arange(half_edges.shape[0])
    previous = half_edges[:, 0]
    following = half_edges[:, 1]
    return jnp.all(half_edges[previous, 1] == half_edge_ids) & jnp.all(
        half_edges[following, 0] == half_edge_ids
    )


@partial(
    jax.jit,
    static_argnames=("width", "height", "energy_fn", "min_face_vertices_after_t1"),
)
def update_t1(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    vertex_params: Array,
    half_edge_params: Array,
    face_params: Array,
    energy_fn: InnerEnergy,
    min_dist_t1: float,
    selected_vertices: Array,
    selected_half_edges: Array,
    selected_faces: Array,
    t1_invalid_degree_penalty: float,
    min_face_vertices_after_t1: int,
) -> tuple[Array, Array, Array]:
    """Detect and apply periodic T1 transitions inside ``cvm3d``.

    The updater scans one representative from each twin half-edge pair. If a
    non-triangular interface is shorter than ``min_dist_t1``, it compares a
    topology-rotated T1 candidate against a same-topology shortened candidate
    and keeps the lower-energy state.
    """
    del selected_vertices, selected_half_edges, selected_faces

    def loss(
        current_vertices: Array, current_half_edges: Array, current_faces: Array
    ) -> Array:
        return energy_fn(
            current_vertices,
            current_half_edges,
            current_faces,
            vertex_params,
            half_edge_params,
            face_params,
        )

    def body_fun(
        edge_pair_id: Array, state: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        he_idx = 2 * edge_pair_id
        current_vertices, current_half_edges, current_faces = state

        he = current_half_edges.at[he_idx].get()
        prev_he = current_half_edges.at[he[0]].get()
        next_he = current_half_edges.at[he[1]].get()
        twin_he = current_half_edges.at[he[2]].get()
        twin_prev_he = current_half_edges.at[prev_he[2]].get()
        next_twin_he = current_half_edges.at[twin_he[1]].get()
        twin_next_he = current_half_edges.at[next_he[2]].get()

        prev_he_idx = he.at[0].get()
        next_he_idx = he.at[1].get()
        twin_he_idx = he.at[2].get()
        twin_prev_he_idx = prev_he.at[2].get()
        prev_twin_prev_he_idx = twin_prev_he.at[0].get()
        prev_twin_he_idx = twin_he.at[0].get()
        next_twin_he_idx = twin_he.at[1].get()
        twin_next_he_idx = next_he.at[2].get()
        next_twin_next_he_idx = twin_next_he.at[1].get()

        distance = edge_length(
            he_idx, current_vertices, current_half_edges, width, height
        )
        should_update = current_half_edges.at[prev_he_idx, 0].get() != next_he_idx
        twin_should_update = (
            current_half_edges.at[prev_twin_he_idx, 0].get() != next_twin_he_idx
        )

        def update_state(
            candidate_state: tuple[Array, Array, Array],
        ) -> tuple[Array, Array, Array]:
            candidate_vertices, candidate_half_edges, candidate_faces = candidate_state

            t1_half_edges = candidate_half_edges
            t1_half_edges = t1_half_edges.at[he_idx, 0].set(prev_twin_prev_he_idx)
            t1_half_edges = t1_half_edges.at[he_idx, 1].set(twin_prev_he_idx)
            t1_half_edges = t1_half_edges.at[he_idx, 5].set(twin_prev_he[5])

            t1_half_edges = t1_half_edges.at[twin_he_idx, 0].set(twin_next_he_idx)
            t1_half_edges = t1_half_edges.at[twin_he_idx, 1].set(next_twin_next_he_idx)
            t1_half_edges = t1_half_edges.at[twin_he_idx, 5].set(twin_next_he[5])

            t1_half_edges = t1_half_edges.at[prev_he_idx, 1].set(next_he_idx)
            t1_half_edges = t1_half_edges.at[prev_he_idx, 4].set(he[4])
            t1_half_edges = t1_half_edges.at[prev_he_idx, 6].add(he[6])
            t1_half_edges = t1_half_edges.at[prev_he_idx, 7].add(he[7])

            t1_half_edges = t1_half_edges.at[next_he_idx, 0].set(prev_he_idx)

            t1_half_edges = t1_half_edges.at[prev_twin_he_idx, 1].set(next_twin_he_idx)
            t1_half_edges = t1_half_edges.at[prev_twin_he_idx, 4].set(twin_he[4])
            t1_half_edges = t1_half_edges.at[prev_twin_he_idx, 6].add(twin_he[6])
            t1_half_edges = t1_half_edges.at[prev_twin_he_idx, 7].add(twin_he[7])

            t1_half_edges = t1_half_edges.at[next_twin_he_idx, 0].set(prev_twin_he_idx)
            t1_half_edges = t1_half_edges.at[prev_twin_prev_he_idx, 1].set(he_idx)

            t1_half_edges = t1_half_edges.at[twin_prev_he_idx, 0].set(he_idx)
            t1_half_edges = t1_half_edges.at[twin_prev_he_idx, 3].set(he[4])
            t1_half_edges = t1_half_edges.at[twin_prev_he_idx, 6].add(-he[6])
            t1_half_edges = t1_half_edges.at[twin_prev_he_idx, 7].add(-he[7])

            t1_half_edges = t1_half_edges.at[twin_next_he_idx, 1].set(twin_he_idx)
            t1_half_edges = t1_half_edges.at[next_twin_next_he_idx, 0].set(twin_he_idx)
            t1_half_edges = t1_half_edges.at[next_twin_next_he_idx, 3].set(twin_he[4])
            t1_half_edges = t1_half_edges.at[next_twin_next_he_idx, 6].add(-twin_he[6])
            t1_half_edges = t1_half_edges.at[next_twin_next_he_idx, 7].add(-twin_he[7])

            t1_vertices = _set_t1_vertex_positions(
                candidate_vertices,
                t1_half_edges,
                he_idx,
                twin_he_idx,
                width,
                height,
                min_dist_t1,
                jnp.pi / 2.0,
            )

            t1_faces = candidate_faces
            t1_faces = t1_faces.at[prev_he[5]].set(prev_he_idx)
            t1_faces = t1_faces.at[twin_prev_he[5]].set(twin_prev_he_idx)
            t1_faces = t1_faces.at[next_twin_he[5]].set(next_twin_he_idx)
            t1_faces = t1_faces.at[twin_next_he[5]].set(twin_next_he_idx)

            t1_vertices, t1_half_edges, t1_faces = update_pbc(
                t1_vertices, t1_half_edges, t1_faces, width, height
            )
            loss_t1 = loss(t1_vertices, t1_half_edges, t1_faces)
            valid_t1 = _faces_close_within_max_edges(
                t1_half_edges, t1_faces, max_edges=half_edges.shape[0]
            )
            valid_degrees = _faces_have_minimum_vertices(
                t1_half_edges,
                t1_faces,
                max_edges=half_edges.shape[0],
                min_vertices=min_face_vertices_after_t1,
            )
            valid_links = _half_edge_links_are_consistent(t1_half_edges)
            loss_t1 = loss_t1 + jnp.where(valid_degrees, 0.0, t1_invalid_degree_penalty)

            no_t1_vertices = _set_t1_vertex_positions(
                candidate_vertices,
                candidate_half_edges,
                he_idx,
                twin_he_idx,
                width,
                height,
                min_dist_t1,
                jnp.array(0.0),
            )
            no_t1_vertices, no_t1_half_edges, no_t1_faces = update_pbc(
                no_t1_vertices, candidate_half_edges, candidate_faces, width, height
            )
            loss_no_t1 = loss(no_t1_vertices, no_t1_half_edges, no_t1_faces)

            return jax.lax.cond(
                valid_t1 & valid_degrees & valid_links & (loss_t1 <= loss_no_t1),
                lambda _: (t1_vertices, t1_half_edges, t1_faces),
                lambda _: (no_t1_vertices, no_t1_half_edges, no_t1_faces),
                None,
            )

        return jax.lax.cond(
            (distance <= min_dist_t1) & should_update & twin_should_update,
            update_state,
            lambda candidate_state: candidate_state,
            (current_vertices, current_half_edges, current_faces),
        )

    return jax.lax.fori_loop(
        0, half_edges.shape[0] // 2, body_fun, (vertices, half_edges, faces)
    )
