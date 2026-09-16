"""Differentiable simulation loops for the prismatic 3D CVM."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
import warnings

import jax
import jax.numpy as jnp
import optax
from jax import Array

from cvm3d.geometry import has_self_intersections, update_pbc
from cvm3d.mesh import PrismaticMesh3D
from cvm3d.topology import do_not_update_t1, update_t1 as internal_update_t1

__all__ = ["simulate"]

InnerEnergy = Callable[[Array, Array, Array, Array, Array, Array], Array]
UpdateT1Function = Callable[
    [
        Array,
        Array,
        Array,
        float,
        float,
        Array,
        Array,
        Array,
        InnerEnergy,
        float,
        Array,
        Array,
        Array,
        float,
        int,
    ],
    tuple[Array, Array, Array],
]


def regular_hex_side_length_from_cell_area(cell_area: float) -> float:
    """Return the side length of a regular hexagon with area ``cell_area``."""
    if cell_area <= 0.0:
        msg = "cell_area must be positive."
        raise ValueError(msg)
    return math.sqrt(2.0 * float(cell_area) / (3.0 * math.sqrt(3.0)))


def t1_cutoff_from_scale(
    width: float, height: float, face_count: int, t1_cutoff_scale: float
) -> float:
    """Return the T1 cutoff from periodic-box area per cell and a scale."""
    if face_count <= 0:
        msg = "face_count must be positive."
        raise ValueError(msg)
    if t1_cutoff_scale < 0.0:
        msg = "t1_cutoff_scale must be non-negative."
        raise ValueError(msg)
    cell_area = float(width) * float(height) / int(face_count)
    return regular_hex_side_length_from_cell_area(cell_area) * float(t1_cutoff_scale)


def _resolve_min_dist_t1(
    min_dist_t1: float,
    t1_cutoff_scale: float | None,
    width: float,
    height: float,
    face_count: int,
) -> float:
    if t1_cutoff_scale is None:
        if min_dist_t1 < 0.0:
            msg = "min_dist_t1 must be non-negative."
            raise ValueError(msg)
        return float(min_dist_t1)
    return t1_cutoff_from_scale(width, height, face_count, t1_cutoff_scale)


@partial(
    jax.jit,
    static_argnames=(
        "width",
        "height",
        "energy_fn",
        "solver",
        "iterations",
        "patience",
        "use_gradient_convergence",
        "update_t1_func",
        "reject_self_intersections",
        "max_edges",
        "min_face_vertices_after_t1",
    ),
)
def _jit_minimize_vertices(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    opt_state: optax.OptState,
    previous_losses: Array,
    stagnation: Array,
    vertex_params: Array,
    half_edge_params: Array,
    face_params: Array,
    *,
    width: float,
    height: float,
    energy_fn: InnerEnergy,
    solver: optax.GradientTransformation,
    min_dist_t1: float,
    iterations: int,
    tolerance: float,
    patience: int,
    gradient_tolerance: float,
    use_gradient_convergence: bool,
    update_t1_func: UpdateT1Function,
    reject_self_intersections: bool,
    max_edges: int,
    t1_invalid_degree_penalty: float,
    min_face_vertices_after_t1: int,
) -> tuple[
    tuple[Array, Array, Array], tuple[Array, Array, optax.OptState, Array, Array, Array]
]:
    """JIT-compiled unrolled vertex minimization."""
    loss_history = jnp.zeros((iterations,))
    selected_vertices = jnp.arange(vertices.shape[0])
    selected_half_edges = jnp.arange(half_edges.shape[0])
    selected_faces = jnp.arange(faces.shape[0])

    def loss_for_vertices(
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

    def scan_step(
        carry: tuple[
            Array,
            Array,
            Array,
            optax.OptState,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
        step_id: Array,
    ) -> tuple[
        tuple[
            Array,
            Array,
            Array,
            optax.OptState,
            Array,
            Array,
            Array,
            Array,
            Array,
            Array,
        ],
        None,
    ]:
        (
            current_vertices,
            current_half_edges,
            current_faces,
            current_opt_state,
            prev,
            stagnation,
            stopped,
            converged,
            steps,
            hist,
        ) = carry
        running = jnp.logical_not(stopped)
        loss_value = loss_for_vertices(
            current_vertices, current_half_edges, current_faces
        )
        gradients = jax.grad(loss_for_vertices)(
            current_vertices, current_half_edges, current_faces
        )
        gradient_max_abs = jnp.max(jnp.abs(gradients))
        denom = jnp.where(jnp.abs(prev[0]) > 0, prev[0], 1.0)
        relative_change = jnp.abs((loss_value - prev[0]) / denom)
        next_stagnation = jnp.where(relative_change < tolerance, stagnation + 1, 0)
        if use_gradient_convergence:
            next_converged = gradient_max_abs <= gradient_tolerance
        else:
            next_converged = next_stagnation >= patience
        next_stopped = next_converged | (step_id >= iterations - 1)
        next_converged_flag = converged | (running & next_converged)
        if use_gradient_convergence:
            should_update = running & jnp.logical_not(next_converged)
        else:
            should_update = running

        updates, next_opt_state = solver.update(
            gradients, current_opt_state, current_vertices
        )
        updated_vertices = optax.apply_updates(current_vertices, updates)
        updated_vertices, updated_half_edges, updated_faces = update_pbc(
            updated_vertices, current_half_edges, current_faces, width, height
        )
        updated_vertices, updated_half_edges, updated_faces = update_t1_func(
            updated_vertices,
            updated_half_edges,
            updated_faces,
            width,
            height,
            vertex_params,
            half_edge_params,
            face_params,
            energy_fn,
            min_dist_t1,
            selected_vertices,
            selected_half_edges,
            selected_faces,
            t1_invalid_degree_penalty,
            min_face_vertices_after_t1,
        )
        if reject_self_intersections:
            valid_geometry = jnp.logical_not(
                has_self_intersections(
                    updated_vertices,
                    updated_half_edges,
                    updated_faces,
                    width,
                    height,
                    max_edges=max_edges,
                )
            )
            updated_vertices = jnp.where(
                valid_geometry, updated_vertices, current_vertices
            )
            updated_half_edges = jnp.where(
                valid_geometry, updated_half_edges, current_half_edges
            )
            updated_faces = jnp.where(valid_geometry, updated_faces, current_faces)
            next_opt_state = jax.tree_util.tree_map(
                lambda accepted, rejected: jnp.where(
                    valid_geometry, accepted, rejected
                ),
                next_opt_state,
                current_opt_state,
            )

        current_vertices = jax.lax.cond(
            should_update, lambda: updated_vertices, lambda: current_vertices
        )
        current_half_edges = jax.lax.cond(
            should_update, lambda: updated_half_edges, lambda: current_half_edges
        )
        current_faces = jax.lax.cond(
            should_update, lambda: updated_faces, lambda: current_faces
        )
        current_opt_state = jax.lax.cond(
            should_update, lambda: next_opt_state, lambda: current_opt_state
        )
        stagnation = jax.lax.cond(running, lambda: next_stagnation, lambda: stagnation)
        stopped = jax.lax.cond(running, lambda: next_stopped, lambda: stopped)
        converged = jax.lax.cond(
            running, lambda: next_converged_flag, lambda: converged
        )
        steps = jax.lax.cond(running, lambda: step_id + 1, lambda: steps)

        shifted_prev = prev.at[1:].set(prev[:-1])
        shifted_prev = shifted_prev.at[0].set(loss_value)
        prev = jax.lax.cond(running, lambda: shifted_prev, lambda: prev)
        hist = hist.at[step_id].set(loss_value)
        return (
            current_vertices,
            current_half_edges,
            current_faces,
            current_opt_state,
            prev,
            stagnation,
            stopped,
            converged,
            steps,
            hist,
        ), None

    initial_carry = (
        vertices,
        half_edges,
        faces,
        opt_state,
        previous_losses,
        stagnation,
        jnp.array(False),
        jnp.array(False),
        jnp.array(0),
        loss_history,
    )
    final_state, _ = jax.lax.scan(scan_step, initial_carry, xs=jnp.arange(iterations))
    (
        final_vertices,
        final_half_edges,
        final_faces,
        final_opt_state,
        prev,
        final_stagnation,
        _,
        converged,
        steps,
        hist,
    ) = final_state
    return (
        final_vertices,
        final_half_edges,
        final_faces,
    ), (
        hist,
        steps,
        final_opt_state,
        prev,
        final_stagnation,
        converged,
    )


def minimize_vertices(
    vertices: Array,
    half_edges: Array,
    faces: Array,
    width: float,
    height: float,
    vertex_params: Array,
    half_edge_params: Array,
    face_params: Array,
    energy_fn: InnerEnergy,
    solver: optax.GradientTransformation | None = None,
    min_dist_t1: float = 0.005,
    t1_cutoff_scale: float | None = None,
    iterations: int = 1000,
    tolerance: float = 1e-5,
    patience: int = 10,
    gradient_tolerance: float | None = None,
    update_t1: bool = False,
    reject_self_intersections: bool = False,
    t1_invalid_degree_penalty: float = 1e12,
    min_face_vertices_after_t1: int = 3,
    extend_until_converged: bool = True,
    extension_iterations: int = 1000,
    max_iterations: int | None = None,
    max_edges: int = 20,
) -> tuple[tuple[Array, Array, Array], Array]:
    """Minimize an inner energy over vertex positions.

    The loop is unrolled with ``jax.lax.scan``, so it can be differentiated
    with respect to energy parameters in the usual JAX way. Optional T1 uses
    ``cvm3d``'s internal periodic topology updater. If ``t1_cutoff_scale`` is
    provided, the T1 cutoff is computed as the side length of a regular hexagon
    with area ``width * height / nb_faces`` multiplied by that scale; otherwise
    the absolute ``min_dist_t1`` is used. If ``reject_self_intersections=True``,
    optimizer steps that would make any cell polygon self-intersect are not
    accepted. ``t1_invalid_degree_penalty`` is added to any T1 candidate that
    would leave a face with fewer than ``min_face_vertices_after_t1`` sides.
    If ``gradient_tolerance`` is not ``None``, convergence is detected by
    ``max(abs(dE/du)) <= gradient_tolerance`` instead of energy stagnation.
    """
    solver = optax.sgd(learning_rate=0.01) if solver is None else solver
    update_t1_func = internal_update_t1 if update_t1 else do_not_update_t1
    chunk_iterations = int(iterations)
    if chunk_iterations <= 0:
        msg = "iterations must be a positive integer."
        raise ValueError(msg)
    if int(patience) <= 0:
        msg = "patience must be a positive integer."
        raise ValueError(msg)
    use_gradient_convergence = gradient_tolerance is not None
    effective_gradient_tolerance = (
        0.0 if gradient_tolerance is None else float(gradient_tolerance)
    )
    if effective_gradient_tolerance < 0.0:
        msg = "gradient_tolerance must be non-negative."
        raise ValueError(msg)
    extension_iterations = int(extension_iterations)
    if extension_iterations <= 0:
        msg = "extension_iterations must be a positive integer."
        raise ValueError(msg)
    if max_iterations is not None and int(max_iterations) < chunk_iterations:
        msg = "max_iterations must be None or greater than or equal to iterations."
        raise ValueError(msg)
    if int(min_face_vertices_after_t1) < 3:
        msg = "min_face_vertices_after_t1 must be at least 3."
        raise ValueError(msg)
    if float(t1_invalid_degree_penalty) < 0.0:
        msg = "t1_invalid_degree_penalty must be non-negative."
        raise ValueError(msg)

    effective_min_dist_t1 = _resolve_min_dist_t1(
        min_dist_t1,
        t1_cutoff_scale,
        width,
        height,
        int(faces.shape[0]),
    )

    current_vertices = vertices
    current_half_edges = half_edges
    current_faces = faces
    opt_state = solver.init(vertices)
    initial_loss = energy_fn(
        vertices, half_edges, faces, vertex_params, half_edge_params, face_params
    )
    previous_losses = jnp.full((int(patience),), initial_loss)
    stagnation = jnp.array(0)
    histories: list[Array] = []
    total_steps = 0
    converged = False

    while True:
        remaining = (
            None if max_iterations is None else int(max_iterations) - total_steps
        )
        if remaining is not None and remaining <= 0:
            break
        current_chunk_iterations = (
            chunk_iterations if not histories else extension_iterations
        )
        if remaining is not None:
            current_chunk_iterations = min(current_chunk_iterations, remaining)

        (
            current_vertices,
            current_half_edges,
            current_faces,
        ), (
            history,
            steps,
            opt_state,
            previous_losses,
            stagnation,
            chunk_converged,
        ) = jax.block_until_ready(
            _jit_minimize_vertices(
                current_vertices,
                current_half_edges,
                current_faces,
                opt_state,
                previous_losses,
                stagnation,
                vertex_params,
                half_edge_params,
                face_params,
                width=float(width),
                height=float(height),
                energy_fn=energy_fn,
                solver=solver,
                min_dist_t1=float(effective_min_dist_t1),
                iterations=int(current_chunk_iterations),
                tolerance=float(tolerance),
                patience=int(patience),
                gradient_tolerance=effective_gradient_tolerance,
                use_gradient_convergence=use_gradient_convergence,
                update_t1_func=update_t1_func,
                reject_self_intersections=bool(reject_self_intersections),
                max_edges=int(max_edges),
                t1_invalid_degree_penalty=float(t1_invalid_degree_penalty),
                min_face_vertices_after_t1=int(min_face_vertices_after_t1),
            )
        )
        step_count = int(steps)
        histories.append(history[:step_count])
        total_steps += step_count
        converged = bool(chunk_converged)

        if converged or not extend_until_converged:
            break
        if step_count < current_chunk_iterations:
            break

    if (
        extend_until_converged
        and not converged
        and max_iterations is not None
        and total_steps >= int(max_iterations)
    ):
        warnings.warn(
            "Optimization reached max_iterations before satisfying the convergence criterion.",
            RuntimeWarning,
            stacklevel=2,
        )

    full_history = jnp.concatenate(histories) if histories else jnp.array([])
    return (current_vertices, current_half_edges, current_faces), full_history


def simulate(
    mesh: PrismaticMesh3D,
    energy_fn: InnerEnergy,
    solver: optax.GradientTransformation | None = None,
    min_dist_t1: float = 0.005,
    t1_cutoff_scale: float | None = None,
    iterations: int = 1000,
    tolerance: float = 1e-5,
    patience: int = 10,
    gradient_tolerance: float | None = None,
    update_t1: bool = False,
    reject_self_intersections: bool = False,
    t1_invalid_degree_penalty: float = 1e12,
    min_face_vertices_after_t1: int = 3,
    extend_until_converged: bool = True,
    extension_iterations: int = 1000,
    max_iterations: int | None = None,
) -> Array:
    """Run an in-place forward simulation on a ``PrismaticMesh3D``.

    ``t1_cutoff_scale`` takes precedence over the absolute ``min_dist_t1`` when
    it is not ``None``. ``reject_self_intersections=True`` rejects optimizer
    updates that would create a self-intersecting cell polygon. T1 candidates
    that would leave a face with fewer than ``min_face_vertices_after_t1`` sides
    receive ``t1_invalid_degree_penalty``. If ``gradient_tolerance`` is not
    ``None``, convergence is detected by force balance,
    ``max(abs(dE/du)) <= gradient_tolerance``.
    """
    (mesh.vertices, mesh.half_edges, mesh.faces), history = minimize_vertices(
        mesh.vertices,
        mesh.half_edges,
        mesh.faces,
        mesh.width,
        mesh.height,
        mesh.vertices_params,
        mesh.half_edges_params,
        mesh.faces_params,
        energy_fn,
        solver=solver,
        min_dist_t1=min_dist_t1,
        t1_cutoff_scale=t1_cutoff_scale,
        iterations=iterations,
        tolerance=tolerance,
        patience=patience,
        gradient_tolerance=gradient_tolerance,
        update_t1=update_t1,
        reject_self_intersections=reject_self_intersections,
        t1_invalid_degree_penalty=t1_invalid_degree_penalty,
        min_face_vertices_after_t1=min_face_vertices_after_t1,
        extend_until_converged=extend_until_converged,
        extension_iterations=extension_iterations,
        max_iterations=max_iterations,
        max_edges=mesh.max_edges,
    )
    return history
