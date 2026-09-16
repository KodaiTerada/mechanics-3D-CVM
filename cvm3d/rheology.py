"""Linear frequency response of the public prismatic model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from cvm3d.energy import energy_from_tables
from cvm3d.mesh import PrismaticMesh3D

RheologyMode = Literal["bulk", "shear", "simple_shear"]


@dataclass(frozen=True)
class ComplexModulusResult:
    """Fixed-topology linear response for one imposed strain mode."""

    mode: RheologyMode
    angular_frequencies: NDArray[np.float64]
    complex_modulus: NDArray[np.complex128]
    affine_modulus: float
    relaxed_modulus: float | None
    relaxation_rates: NDArray[np.float64]
    hessian_eigenvalues: NDArray[np.float64]
    strain_couplings: NDArray[np.float64]
    mode_shapes: NDArray[np.float64]
    energy: float
    gradient_norm: float
    gradient_max_abs: float

    @property
    def storage_modulus(self) -> NDArray[np.float64]:
        """Return the in-phase elastic response."""
        return np.asarray(self.complex_modulus.real, dtype=np.float64)

    @property
    def loss_modulus(self) -> NDArray[np.float64]:
        """Return the out-of-phase dissipative response."""
        return np.asarray(self.complex_modulus.imag, dtype=np.float64)


def _translation_free_basis(vertex_count: int) -> NDArray[np.float64]:
    """Return an orthonormal basis excluding rigid x/y translations."""
    if vertex_count <= 0:
        raise ValueError("mesh must contain at least one vertex.")
    helmert = np.zeros((vertex_count, max(vertex_count - 1, 0)), dtype=np.float64)
    for column in range(vertex_count - 1):
        count = column + 1
        normalization = np.sqrt(count * (count + 1.0))
        helmert[:count, column] = 1.0 / normalization
        helmert[count, column] = -count / normalization
    return np.kron(helmert, np.eye(2, dtype=np.float64))


def _validate_frequencies(
    angular_frequencies: NDArray[np.float64] | list[float],
) -> NDArray[np.float64]:
    frequencies = np.asarray(angular_frequencies, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError(
            "angular_frequencies must be a non-empty one-dimensional array."
        )
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("angular_frequencies must contain only finite values.")
    if np.any(frequencies < 0.0):
        raise ValueError("angular_frequencies must be non-negative.")
    return frequencies


def _validate_mode(mode: RheologyMode) -> None:
    if mode not in {"bulk", "shear", "simple_shear"}:
        raise ValueError(
            "mode must be 'bulk', 'shear' (pure shear), or 'simple_shear'."
        )


def _response_vertices(vertices: Array) -> Array:
    """Promote response derivatives to float64 when JAX x64 is enabled."""
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.asarray(vertices).dtype
    return jnp.asarray(vertices, dtype=dtype)


def _strain_scales(mode: RheologyMode, strain: Array) -> Array:
    if mode == "bulk":
        scale_x = jnp.exp(0.5 * strain)
        scale_y = scale_x
    else:
        scale_x = jnp.exp(0.5 * strain)
        scale_y = jnp.exp(-0.5 * strain)
    return jnp.stack((scale_x, scale_y))


def complex_modulus(
    mesh: PrismaticMesh3D,
    angular_frequencies: NDArray[np.float64] | list[float],
    *,
    mode: RheologyMode,
    mobility: float = 1.0,
    edge_star_strength: float = 1.0,
    eigenvalue_tolerance: float = 1e-5,
    gradient_tolerance: float = 1e-3,
    zero_mode_coupling_tolerance: float = 1e-7,
    require_equilibrium: bool = True,
    target_cell_volumes: Array | float | None = None,
    volume_penalty: Array | float = 0.0,
) -> ComplexModulusResult:
    r"""Return the fixed-topology linear complex bulk or shear modulus.

    The overdamped response is calculated from the translation-free Hessian
    ``H``, strain coupling ``Xi``, and affine stiffness ``C`` as

    ``M*(omega) = (C - Xi.T @ (H + i omega / mobility)^-1 @ Xi) / area``.

    ``shear`` uses F=diag(exp(strain/2), exp(-strain/2));
    ``simple_shear`` uses F=[[1, strain], [0, 1]], with engineering shear
    strain. Both are differentiated at zero strain; this is linear response,
    not a finite-strain shear trajectory. Nonaffine displacements are Cartesian.

    The mesh face table supplies ``gamma_b``, ``gamma_c``, ``sigma``, and an
    optional fourth column of local cell volumes. The topology is held fixed.
    """
    frequencies = _validate_frequencies(angular_frequencies)
    _validate_mode(mode)
    if not np.isfinite(mobility) or mobility <= 0.0:
        raise ValueError("mobility must be finite and positive.")
    if eigenvalue_tolerance < 0.0:
        raise ValueError("eigenvalue_tolerance must be non-negative.")
    if gradient_tolerance < 0.0:
        raise ValueError("gradient_tolerance must be non-negative.")
    if zero_mode_coupling_tolerance < 0.0:
        raise ValueError("zero_mode_coupling_tolerance must be non-negative.")

    reference_vertices = _response_vertices(mesh.vertices)
    if reference_vertices.ndim != 2 or reference_vertices.shape[1] != 2:
        raise ValueError(
            "complex_modulus currently requires mesh.vertices with shape (n, 2)."
        )
    if not jnp.issubdtype(reference_vertices.dtype, jnp.inexact):
        raise TypeError("mesh.vertices must have a floating-point dtype.")

    vertex_shape = reference_vertices.shape
    zero_displacement = jnp.zeros(
        reference_vertices.size, dtype=reference_vertices.dtype
    )
    zero_strain = jnp.asarray(0.0, dtype=reference_vertices.dtype)
    reference_area = float(mesh.width * mesh.height)
    if not np.isfinite(reference_area) or reference_area <= 0.0:
        raise ValueError("mesh width and height must define a finite positive area.")

    def deformed_energy(flat_displacement: Array, strain: Array) -> Array:
        displacement = flat_displacement.reshape(vertex_shape)
        if mode == "simple_shear":
            # Columns are the deformed lattice vectors. Keep integer image
            # offsets fixed, including edges crossing the y boundary.
            deformation = jnp.array([[1.0, strain], [0.0, 1.0]])
            scale_x = scale_y = 1.0
            vertices = reference_vertices @ deformation.T + displacement
            box_matrix = deformation @ jnp.diag(jnp.array([mesh.width, mesh.height]))
        else:
            scales = _strain_scales(mode, strain)
            scale_x, scale_y = scales
            vertices = reference_vertices * scales + displacement
            box_matrix = None
        return energy_from_tables(
            vertices,
            mesh.half_edges,
            mesh.faces,
            mesh.vertices_params,
            mesh.half_edges_params,
            mesh.faces_params,
            box_matrix=box_matrix,
            width=mesh.width * scale_x,
            height=mesh.height * scale_y,
            total_volume=mesh.total_volume,
            edge_star_strength=edge_star_strength,
            max_edges=mesh.max_edges,
            target_cell_volumes=target_cell_volumes,
            volume_penalty=volume_penalty,
        )

    energy_value, gradient = jax.value_and_grad(deformed_energy, argnums=0)(
        zero_displacement,
        zero_strain,
    )
    hessian = jax.hessian(
        lambda displacement: deformed_energy(displacement, zero_strain)
    )(zero_displacement)
    strain_coupling = jax.jacfwd(
        jax.grad(deformed_energy, argnums=0),
        argnums=1,
    )(zero_displacement, zero_strain)
    affine_stiffness = jax.grad(
        jax.grad(lambda strain: deformed_energy(zero_displacement, strain))
    )(zero_strain)

    energy_number = float(np.asarray(energy_value))
    gradient_array = np.asarray(gradient, dtype=np.float64)
    hessian_array = np.asarray(hessian, dtype=np.float64)
    coupling_array = np.asarray(strain_coupling, dtype=np.float64)
    affine_stiffness_number = float(np.asarray(affine_stiffness))
    if not all(
        (
            np.isfinite(energy_number),
            np.all(np.isfinite(gradient_array)),
            np.all(np.isfinite(hessian_array)),
            np.all(np.isfinite(coupling_array)),
            np.isfinite(affine_stiffness_number),
        )
    ):
        raise ValueError("linear response derivatives must all be finite.")

    gradient_norm = float(np.linalg.norm(gradient_array))
    gradient_max_abs = float(np.max(np.abs(gradient_array)))
    if require_equilibrium and gradient_max_abs > gradient_tolerance:
        raise ValueError(
            "mesh is not at mechanical equilibrium: "
            f"max |dE/du|={gradient_max_abs:.6g} exceeds "
            f"gradient_tolerance={gradient_tolerance:.6g}."
        )

    basis = _translation_free_basis(vertex_shape[0])
    projected_hessian = basis.T @ (0.5 * (hessian_array + hessian_array.T)) @ basis
    projected_coupling = basis.T @ coupling_array
    eigenvalues, eigenvectors = np.linalg.eigh(projected_hessian)
    if eigenvalues.size and eigenvalues[0] < -eigenvalue_tolerance:
        raise ValueError(
            "mesh is linearly unstable: "
            f"minimum translation-free Hessian eigenvalue is {eigenvalues[0]:.6g}."
        )

    mode_couplings = eigenvectors.T @ projected_coupling
    mode_shapes = (basis @ eigenvectors).T.reshape((-1, *vertex_shape))
    zero_modes = eigenvalues <= eigenvalue_tolerance
    coupling_scale = max(float(np.linalg.norm(mode_couplings)), 1.0)
    coupled_zero_modes = zero_modes & (
        np.abs(mode_couplings) > zero_mode_coupling_tolerance * coupling_scale
    )
    if np.any(coupled_zero_modes) and np.any(frequencies == 0.0):
        raise ValueError(
            "zero-frequency response is undefined because strain couples to a zero Hessian mode."
        )

    stable_eigenvalues = np.where(zero_modes, 0.0, eigenvalues)
    friction = 1.0 / float(mobility)
    response = np.empty(frequencies.shape, dtype=np.complex128)
    for index, frequency in enumerate(frequencies):
        if frequency == 0.0:
            active = ~zero_modes
            relaxation = np.sum(
                mode_couplings[active] ** 2 / stable_eigenvalues[active]
            )
        else:
            denominator = stable_eigenvalues + 1j * frequency * friction
            relaxation = np.sum(mode_couplings**2 / denominator)
        response[index] = (affine_stiffness_number - relaxation) / reference_area

    if np.any(coupled_zero_modes):
        relaxed_modulus = None
    else:
        active = ~zero_modes
        relaxed_stiffness = affine_stiffness_number - np.sum(
            mode_couplings[active] ** 2 / stable_eigenvalues[active]
        )
        relaxed_modulus = float(relaxed_stiffness / reference_area)

    return ComplexModulusResult(
        mode=mode,
        angular_frequencies=frequencies.copy(),
        complex_modulus=response,
        affine_modulus=float(affine_stiffness_number / reference_area),
        relaxed_modulus=relaxed_modulus,
        relaxation_rates=np.asarray(
            float(mobility) * stable_eigenvalues, dtype=np.float64
        ),
        hessian_eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        strain_couplings=np.asarray(mode_couplings, dtype=np.float64),
        mode_shapes=np.asarray(mode_shapes, dtype=np.float64),
        energy=energy_number,
        gradient_norm=gradient_norm,
        gradient_max_abs=gradient_max_abs,
    )


__all__ = ["ComplexModulusResult", "complex_modulus"]
