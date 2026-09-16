"""Guarded Adam/BFGS relaxation with a force-based convergence check."""

from dataclasses import dataclass
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from cvm3d.geometry import has_self_intersections, update_pbc
from cvm3d.rheology import _translation_free_basis


@dataclass(frozen=True)
class RelaxationResult:
    mesh: object
    energy_history: np.ndarray
    gradient_max_abs: float
    converged: bool
    stages: tuple[str, ...]


def relax_fixed_topology(
    mesh,
    energy: Callable,
    *,
    gradient_tolerance: float = 5e-5,
    adam_iterations: int = 500,
    adam_learning_rate: float = 5e-5,
    bfgs_max_iterations: int = 10000,
    max_retries: int = 3,
    newton_iterations: int = 12,
    progress: Callable[[str, float], None] | None = None,
) -> RelaxationResult:
    """Return a relaxed copy; preserve topology, targets, and physical energy.

    Every attempt starts with Adam, followed by fresh BFGS. Retries decrease
    Adam's step by a factor of ten. A damped Hessian solve polishes small force
    residuals when energy-based line searches reach floating-point precision.
    Success always requires the specified physical max-force tolerance.
    ``converged=False`` is returned on exhaustion; callers must not cache that
    result as an equilibrium. This routine does not test Hessian stability.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Force-controlled relaxation requires JAX x64 enabled.")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be finite and positive.")
    if not np.isfinite(adam_learning_rate) or adam_learning_rate <= 0:
        raise ValueError("adam_learning_rate must be finite and positive.")
    if min(adam_iterations, bfgs_max_iterations, max_retries, newton_iterations) < 0:
        raise ValueError("Iteration limits must be nonnegative.")
    result_mesh = mesh.copy()
    shape = mesh.vertices.shape
    x = np.asarray(mesh.vertices, dtype=float).ravel().copy()

    def flat_energy(values):
        return energy(
            values.reshape(shape),
            mesh.half_edges,
            mesh.faces,
            mesh.vertices_params,
            mesh.half_edges_params,
            mesh.faces_params,
        )

    value_grad = jax.jit(jax.value_and_grad(flat_energy))
    hessian = jax.jit(jax.hessian(flat_energy))
    valid = jax.jit(
        lambda values: ~has_self_intersections(
            values.reshape(shape),
            mesh.half_edges,
            mesh.faces,
            mesh.width,
            mesh.height,
            max_edges=mesh.max_edges,
        )
    )

    def evaluate(values):
        if not np.all(np.isfinite(values)) or not bool(valid(jnp.asarray(values))):
            return np.inf, np.full_like(values, np.nan)
        value, gradient = value_grad(jnp.asarray(values))
        value, gradient = float(value), np.asarray(gradient, dtype=float)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return np.inf, np.full_like(values, np.nan)
        return value, gradient

    value, gradient = evaluate(x)
    if not np.isfinite(value):
        raise ValueError("Initial geometry or energy is invalid; cannot relax it.")
    history = [value]
    stages = []
    # Limit one accepted displacement to 5% of a regular-cell edge length.
    cell_length = np.sqrt(
        2 * mesh.width * mesh.height / (3 * np.sqrt(3) * mesh.nb_faces)
    )
    max_step = 0.05 * cell_length

    def force(g):
        return float(np.max(np.abs(g)))

    def report(label):
        stages.append(label)
        if progress is not None:
            progress(label, force(gradient))

    def line_step(direction):
        nonlocal x, value, gradient
        # Remove the translational gauge, without modifying physical forces.
        d = direction.reshape(shape).copy()
        d -= d.mean(axis=0)
        d = d.ravel()
        norm = np.max(np.linalg.norm(d.reshape(shape), axis=1))
        if norm > max_step:
            d *= max_step / norm
        slope = float(gradient @ d)
        if not np.isfinite(slope) or slope >= 0:
            return False
        # Only rounding-sized energy differences may be accepted by force
        # reduction instead of Armijo; this is not an energy-stagnation exit.
        rounding = 64 * np.finfo(float).eps * max(1.0, abs(value))
        for _ in range(24):
            new_value, new_gradient = evaluate(x + d)
            if np.isfinite(new_value) and (
                new_value <= value + 1e-4 * slope
                or (
                    abs(new_value - value) <= rounding
                    and force(new_gradient) < 0.9 * force(gradient)
                )
            ):
                x, value, gradient = x + d, new_value, new_gradient
                history.append(value)
                return True
            d *= 0.5
            slope *= 0.5
        return False

    for attempt in range(max_retries + 1):
        if force(gradient) <= gradient_tolerance:
            break
        rate = adam_learning_rate * 0.1**attempt
        first, second = np.zeros_like(x), np.zeros_like(x)
        for step in range(1, adam_iterations + 1):
            if force(gradient) <= gradient_tolerance:
                break
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            direction = (
                -rate
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
            if gradient @ direction >= 0:
                direction = (
                    -rate * gradient / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
                )
            if not line_step(direction):
                break
        report(f"Adam {attempt+1}, lr={rate:.3g}")
        if force(gradient) <= gradient_tolerance:
            break

        # Keep the best valid accepted iterate even if scipy reports failure.
        def accept_iterate(candidate):
            nonlocal x, value, gradient
            new_value, new_gradient = evaluate(candidate)
            rounding = 64 * np.finfo(float).eps * max(1.0, abs(value))
            if np.isfinite(new_value) and (
                new_value < value - rounding
                or (
                    new_value <= value + rounding
                    and force(new_gradient) < force(gradient)
                )
            ):
                x, value, gradient = (
                    np.array(candidate, copy=True),
                    new_value,
                    new_gradient,
                )
                history.append(value)

        optimized = minimize(
            evaluate,
            x.copy(),
            jac=True,
            method="BFGS",
            callback=accept_iterate,
            options={"gtol": gradient_tolerance, "maxiter": bfgs_max_iterations},
        )
        accept_iterate(optimized.x)
        report(f"BFGS {attempt+1}: {optimized.message}")
        if force(gradient) <= gradient_tolerance:
            break

        # Precision-loss exits occur near the minimum. Solve in the subspace
        # without translations, where the physical Hessian can be inverted.
        if force(gradient) <= max(1e-2, 100 * gradient_tolerance):
            basis = _translation_free_basis(shape[0])
            for _ in range(newton_iterations):
                if force(gradient) <= gradient_tolerance:
                    break
                H = np.asarray(hessian(jnp.asarray(x)))
                reduced = basis.T @ ((H + H.T) / 2) @ basis
                if not np.all(np.isfinite(reduced)):
                    break
                eigenvalues, vectors = np.linalg.eigh(reduced)
                # Positive curvature floor makes the polishing step descent
                # even if a small negative/soft mode is present.
                floor = max(1e-8, 1e-8 * np.max(np.abs(eigenvalues)))
                direction = -basis @ (
                    vectors
                    @ (
                        (vectors.T @ (basis.T @ gradient))
                        / np.maximum(eigenvalues, floor)
                    )
                )
                if not line_step(direction):
                    break
            report(f"Hessian polish {attempt+1}")

    result_mesh.vertices, result_mesh.half_edges, result_mesh.faces = update_pbc(
        jnp.asarray(x).reshape(shape),
        mesh.half_edges,
        mesh.faces,
        mesh.width,
        mesh.height,
    )
    final_value, final_gradient = jax.value_and_grad(
        lambda vertices: energy(
            vertices,
            result_mesh.half_edges,
            result_mesh.faces,
            result_mesh.vertices_params,
            result_mesh.half_edges_params,
            result_mesh.faces_params,
        )
    )(result_mesh.vertices)
    residual = force(np.asarray(final_gradient))
    converged = bool(
        np.isfinite(float(final_value))
        and np.isfinite(residual)
        and residual <= gradient_tolerance
    )
    return RelaxationResult(
        result_mesh, np.asarray(history), residual, converged, tuple(stages)
    )
