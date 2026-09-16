"""Fixed-shear Hessian checks for the cell-corner three-edge model.

This module implements the cell-corner model without
identifying the apical and basal footprints.  Every vertex therefore
has independent basal and apical in-plane coordinates.  The two layers remain
planar at ``z=0`` and ``z=h``; this is the smallest fixed-topology extension of
the prismatic model that contains both a homogeneous out-of-plane shear and
all periodic non-affine vertex modes.

The implementation is intentionally limited to the geometry, energy, and
fixed-shear Hessian analysis used by the public Figure 4 notebook.  A quadratic
penalty can constrain either individual cell volumes or only the total tissue
volume.  Rigid layer translations are projected out exactly in the Hessian
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from cvm3d._initializers_core import periodic_voronoi_tables, triangular_lattice_seeds

VolumePenaltyMode = Literal["cell", "tissue"]

__all__ = [
    "FixedShearHessian",
    "CellParameters",
    "ShearMesh",
    "VolumePenaltyMode",
    "affine_shear_energy_difference",
    "affine_shear_magnitude",
    "out_of_plane_modulus",
    "regular_energy",
    "regular_energy_derivative",
]


@dataclass(frozen=True)
class CellParameters:
    """Uniform parameters of the cell-corner three-edge model."""

    gamma_b: float
    gamma_c: float
    sigma: float
    kappa: float
    cell_volume: float = 1.0
    volume_penalty: float = 1.0e4
    volume_penalty_mode: VolumePenaltyMode = "cell"

    def __post_init__(self) -> None:
        if self.volume_penalty_mode not in ("cell", "tissue"):
            raise ValueError("volume_penalty_mode must be 'cell' or 'tissue'.")


@dataclass(frozen=True)
class FixedShearHessian:
    """Hessian spectrum in the non-affine subspace at prescribed shear.

    A fixed macroscopic shear removes the mean displacement of each layer
    separately.  Consequently, both rigid translations and the uniform
    apical--basal relative translations representing additional affine shear
    are absent from this spectrum.  ``eigenvectors`` contains the full-space
    flattened mode vectors as columns.
    """

    energy: float
    gradient_max_abs: float
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    projected_hessian: NDArray[np.float64]
    negative_count: int
    zero_count: int
    positive_count: int
    minimum_eigenvalue: float
    most_unstable_mode: NDArray[np.float64]


def _helmert_zero_mean_basis(size: int) -> NDArray[np.float64]:
    """Return an orthonormal basis for vectors whose entries sum to zero."""
    if size < 2:
        raise ValueError("At least two vertices are required for non-affine analysis.")
    basis = np.zeros((size, size - 1), dtype=np.float64)
    for column in range(size - 1):
        count = column + 1
        basis[:count, column] = 1.0 / math.sqrt(count * (count + 1.0))
        basis[count, column] = -count / math.sqrt(count * (count + 1.0))
    return basis


def regular_energy(
    side_length: float,
    parameters: CellParameters,
) -> float:
    """Return the regular-cell energy."""
    l_value = float(side_length)
    if l_value <= 0.0:
        raise ValueError("side_length must be positive.")
    p = parameters
    root3 = math.sqrt(3.0)
    return (
        1.5 * root3 * p.gamma_b * l_value**2
        + 2.0 * root3 * p.gamma_c * p.cell_volume / (3.0 * l_value)
        + 6.0 * p.sigma * l_value
        + p.kappa * (32.0 / (3.0 * l_value**2) + 27.0 * l_value**4 / p.cell_volume**2)
    )


def regular_energy_derivative(
    side_length: float,
    parameters: CellParameters,
) -> float:
    """Return the side-length derivative of :func:`regular_energy`."""
    l_value = float(side_length)
    p = parameters
    root3 = math.sqrt(3.0)
    return (
        3.0 * root3 * p.gamma_b * l_value
        - 2.0 * root3 * p.gamma_c * p.cell_volume / (3.0 * l_value**2)
        + 6.0 * p.sigma
        - 64.0 * p.kappa / (3.0 * l_value**3)
        + 108.0 * p.kappa * l_value**3 / p.cell_volume**2
    )


def out_of_plane_modulus(side_length: float, parameters: CellParameters) -> float:
    """Return the corner-model affine out-of-plane shear modulus."""
    l_value = float(side_length)
    p = parameters
    return p.gamma_c / (math.sqrt(3.0) * l_value) + 32.0 * p.kappa / (
        3.0 * p.cell_volume * l_value**2
    )


def affine_shear_energy_difference(
    side_length: float,
    shear: float,
    parameters: CellParameters,
) -> float:
    """Return the corner-model affine energy difference per cell for edge-aligned shear."""
    l_value = float(side_length)
    q_value = float(shear)
    height = 2.0 * parameters.cell_volume / (3.0 * math.sqrt(3.0) * l_value**2)
    return 2.0 * parameters.gamma_c * l_value * height * (
        math.sqrt(1.0 + 0.75 * q_value**2) - 1.0
    ) + 16.0 * parameters.kappa * q_value**2 / (3.0 * l_value**2)


def affine_shear_magnitude(side_length: float, parameters: CellParameters) -> float:
    """Return the corner-model positive affine shear solution, or zero."""
    l_value = float(side_length)
    ratio = (
        -3.0
        * parameters.gamma_c
        * parameters.cell_volume
        * l_value
        / (32.0 * math.sqrt(3.0) * parameters.kappa)
    )
    if ratio <= 1.0:
        return 0.0
    return 2.0 * math.sqrt(ratio**2 - 1.0) / math.sqrt(3.0)


@dataclass(frozen=True)
class ShearMesh:
    """Regular periodic double-layer mesh for cell-corner shear verification."""

    reference_xy: Array
    face_vertex_ids: Array
    outgoing_vertex_ids: Array
    width: float
    box_height: float
    layer_height: float
    side_length: float
    cell_volume: float

    @property
    def vertex_count(self) -> int:
        return int(self.reference_xy.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.face_vertex_ids.shape[0])

    @classmethod
    def regular_hexagonal(
        cls,
        nx: int,
        ny: int,
        side_length: float,
        cell_volume: float = 1.0,
    ) -> "ShearMesh":
        """Build a periodic regular-hexagonal physical-vertex mesh."""
        seeds, width, box_height = triangular_lattice_seeds(nx, ny, side_length)
        vertices, half_edges, faces = periodic_voronoi_tables(
            seeds,
            width,
            box_height,
            allowed_face_degrees={6},
            max_edges=20,
        )
        half_edges = np.asarray(half_edges, dtype=np.int32)
        faces = np.asarray(faces, dtype=np.int32)
        face_vertex_ids: list[list[int]] = []
        for start_value in faces:
            current = int(start_value)
            ids: list[int] = []
            for _ in range(20):
                ids.append(int(half_edges[current, 3]))
                current = int(half_edges[current, 1])
                if current == int(start_value):
                    break
            if len(ids) != 6:
                raise ValueError("ShearMesh requires a regular hexagonal topology.")
            face_vertex_ids.append(ids)

        outgoing_ids: list[list[int]] = [[] for _ in range(vertices.shape[0])]
        for half_edge in half_edges:
            source = int(half_edge[3])
            outgoing_ids[source].append(int(half_edge[4]))
        if any(len(ids) != 3 for ids in outgoing_ids):
            raise ValueError("Every physical footprint vertex must have degree three.")

        height = (
            2.0 * float(cell_volume) / (3.0 * math.sqrt(3.0) * float(side_length) ** 2)
        )
        return cls(
            reference_xy=jnp.asarray(vertices),
            face_vertex_ids=jnp.asarray(face_vertex_ids, dtype=jnp.int32),
            outgoing_vertex_ids=jnp.asarray(outgoing_ids, dtype=jnp.int32),
            width=float(width),
            box_height=float(box_height),
            layer_height=height,
            side_length=float(side_length),
            cell_volume=float(cell_volume),
        )

    def zero_displacements(self) -> Array:
        """Return zero basal/apical displacements with shape ``(2, V, 2)``."""
        return jnp.zeros((2, self.vertex_count, 2), dtype=self.reference_xy.dtype)

    def edge_aligned_shear(self, magnitude: float) -> NDArray[np.float64]:
        """Return a shear vector parallel to one regular-hexagon edge.

        The initializer's hexagon edges are at 30 degrees to the global x axis,
        whereas Eqs. (39)--(40) in the notes choose that edge as their x axis.
        This helper makes the convention explicit.
        """
        return float(magnitude) * np.asarray(
            [math.sqrt(3.0) / 2.0, 0.5], dtype=np.float64
        )

    def layer_positions(
        self, displacements: Array, shear: Array | NDArray[np.float64]
    ) -> tuple[Array, Array]:
        """Return basal and apical physical vertex positions in three dimensions."""
        displacement_values = jnp.asarray(displacements).reshape(
            (2, self.vertex_count, 2)
        )
        shear_values = jnp.asarray(shear, dtype=self.reference_xy.dtype).reshape(2)
        basal_xy = self.reference_xy + displacement_values[0]
        apical_xy = (
            self.reference_xy
            + displacement_values[1]
            + self.layer_height * shear_values
        )
        basal = jnp.column_stack(
            (basal_xy, jnp.zeros(self.vertex_count, dtype=self.reference_xy.dtype))
        )
        apical = jnp.column_stack(
            (
                apical_xy,
                jnp.full(
                    self.vertex_count, self.layer_height, dtype=self.reference_xy.dtype
                ),
            )
        )
        return basal, apical

    def _minimum_image_xy(self, differences: Array) -> Array:
        """Return in-plane differences in the nearest periodic image."""
        box = jnp.asarray([self.width, self.box_height], dtype=differences.dtype)
        return differences - box * jnp.round(differences / box)

    def _unfold_faces(self, positions: Array) -> Array:
        """Unfold every face from current minimum-image edge vectors.

        Recomputing the image convention from the current coordinates makes
        the geometry invariant when any stored vertex is shifted by an integer
        number of periodic boxes.
        """
        face_positions = positions[self.face_vertex_ids]
        xy = face_positions[..., :2]
        edge_xy = self._minimum_image_xy(jnp.roll(xy, -1, axis=1) - xy)
        zero = jnp.zeros((*edge_xy.shape[:1], 1, 2), dtype=positions.dtype)
        offsets = jnp.concatenate((zero, jnp.cumsum(edge_xy[:, :-1], axis=1)), axis=1)
        box = jnp.asarray([self.width, self.box_height], dtype=positions.dtype)
        anchor = jnp.mod(xy[:, :1], box)
        return jnp.concatenate((anchor + offsets, face_positions[..., 2:3]), axis=-1)

    def _unfold_layers(self, basal: Array, apical: Array) -> tuple[Array, Array]:
        """Unfold both layers and choose the nearest apico-basal image."""
        bottom = self._unfold_faces(basal)
        top = self._unfold_faces(apical)
        anchor_difference = self._minimum_image_xy(top[:, 0, :2] - bottom[:, 0, :2])
        aligned_top_xy = (
            top[..., :2]
            + (bottom[:, 0, :2] + anchor_difference - top[:, 0, :2])[:, None, :]
        )
        return bottom, jnp.concatenate((aligned_top_xy, top[..., 2:3]), axis=-1)

    def unfold_layers(self, basal: Array, apical: Array) -> tuple[Array, Array]:
        """Return periodically unfolded basal and apical cell-face coordinates."""
        return self._unfold_layers(basal, apical)

    @staticmethod
    def _polygon_area_xy(points: Array) -> Array:
        following = jnp.roll(points, -1, axis=1)
        return 0.5 * jnp.sum(
            points[..., 0] * following[..., 1] - following[..., 0] * points[..., 1],
            axis=1,
        )

    @staticmethod
    def _triangle_area(first: Array, second: Array, third: Array) -> Array:
        return 0.5 * jnp.linalg.norm(jnp.cross(second - first, third - first), axis=-1)

    @staticmethod
    def _triangle_signed_volume(first: Array, second: Array, third: Array) -> Array:
        return jnp.einsum("...i,...i->...", first, jnp.cross(second, third)) / 6.0

    def cell_volumes(
        self, displacements: Array, shear: Array | NDArray[np.float64]
    ) -> Array:
        """Return exact volumes of the straight-sided apico-basal polyhedra."""
        basal, apical = self.layer_positions(displacements, shear)
        bottom, top = self._unfold_layers(basal, apical)
        bottom_next = jnp.roll(bottom, -1, axis=1)
        top_next = jnp.roll(top, -1, axis=1)

        # Fan triangulation of top (CCW) and bottom (CW) faces.
        top_volume = jnp.sum(
            self._triangle_signed_volume(top[:, :1], top[:, 1:-1], top[:, 2:]), axis=1
        )
        bottom_volume = jnp.sum(
            self._triangle_signed_volume(bottom[:, :1], bottom[:, 2:], bottom[:, 1:-1]),
            axis=1,
        )
        side_volume = jnp.sum(
            self._triangle_signed_volume(bottom, bottom_next, top_next)
            + self._triangle_signed_volume(bottom, top_next, top),
            axis=1,
        )
        return top_volume + bottom_volume + side_volume

    def energy_components(
        self,
        displacements: Array,
        shear: Array | NDArray[np.float64],
        parameters: CellParameters,
    ) -> dict[str, Array]:
        """Return physical and selected volume-penalty energy components."""
        basal, apical = self.layer_positions(displacements, shear)
        bottom, top = self._unfold_layers(basal, apical)
        bottom_next = jnp.roll(bottom, -1, axis=1)
        top_next = jnp.roll(top, -1, axis=1)

        basal_areas = self._polygon_area_xy(bottom)
        apical_areas = self._polygon_area_xy(top)
        basal_lengths = jnp.linalg.norm(bottom_next - bottom, axis=-1)
        apical_lengths = jnp.linalg.norm(top_next - top, axis=-1)
        side_areas = self._triangle_area(
            bottom, bottom_next, top_next
        ) + self._triangle_area(bottom, top_next, top)
        basal_energy = parameters.gamma_b * jnp.sum(basal_areas)
        contact_energy = 0.5 * parameters.gamma_c * jnp.sum(side_areas)
        line_energy = parameters.sigma * jnp.sum(apical_lengths)

        def corner_sum(layer, other_layer):
            d1 = jnp.roll(layer, 1, axis=1) - layer
            d2 = jnp.roll(layer, -1, axis=1) - layer
            d3 = other_layer - layer
            tensor = sum(jnp.einsum("...i,...j->...ij", d, d) for d in (d1, d2, d3))
            return jnp.sum(jnp.trace(jnp.linalg.inv(tensor), axis1=-2, axis2=-1)) / 3.0

        star_energy = parameters.kappa * (
            corner_sum(bottom, top) + corner_sum(top, bottom)
        )
        volumes = self.cell_volumes(displacements, shear)
        relative_volume_error = volumes / parameters.cell_volume - 1.0
        relative_tissue_volume_error = jnp.mean(relative_volume_error)
        if parameters.volume_penalty_mode == "cell":
            penalty_energy = (
                0.5 * parameters.volume_penalty * jnp.sum(relative_volume_error**2)
            )
        else:
            # The factor N keeps the penalty extensive and gives the same
            # energy as the cell-wise mode when every cell has the same
            # relative volume error.
            penalty_energy = (
                0.5
                * parameters.volume_penalty
                * relative_volume_error.size
                * relative_tissue_volume_error**2
            )
        physical = basal_energy + contact_energy + line_energy + star_energy
        return {
            "basal": basal_energy,
            "contact": contact_energy,
            "line": line_energy,
            "star": star_energy,
            "physical": physical,
            "volume_penalty": penalty_energy,
            "total": physical + penalty_energy,
            "volumes": volumes,
            "relative_tissue_volume_error": relative_tissue_volume_error,
        }

    def energy(
        self,
        displacements: Array,
        shear: Array | NDArray[np.float64],
        parameters: CellParameters,
    ) -> Array:
        """Return physical plus the selected volume-penalty energy."""
        return self.energy_components(displacements, shear, parameters)["total"]

    def _nonaffine_basis(self) -> NDArray[np.float64]:
        scalar_basis = _helmert_zero_mean_basis(self.vertex_count)
        # Coefficient ordering is (layer, coordinate, zero-mean mode).
        columns: list[NDArray[np.float64]] = []
        for layer in range(2):
            for coordinate in range(2):
                for mode in range(self.vertex_count - 1):
                    vector = np.zeros((2, self.vertex_count, 2), dtype=np.float64)
                    vector[layer, :, coordinate] = scalar_basis[:, mode]
                    columns.append(vector.reshape(-1))
        return np.column_stack(columns)

    def _shared_layer_basis(self) -> NDArray[np.float64]:
        """Return zero-mean modes satisfying identical apical/basal shapes."""
        scalar_basis = _helmert_zero_mean_basis(self.vertex_count)
        columns: list[NDArray[np.float64]] = []
        normalization = math.sqrt(2.0)
        for coordinate in range(2):
            for mode in range(self.vertex_count - 1):
                vector = np.zeros((2, self.vertex_count, 2), dtype=np.float64)
                shared = scalar_basis[:, mode] / normalization
                vector[0, :, coordinate] = shared
                vector[1, :, coordinate] = shared
                columns.append(vector.reshape(-1))
        return np.column_stack(columns)

    def analyze_fixed_shear_hessian(
        self,
        shear: float | NDArray[np.float64],
        parameters: CellParameters,
        *,
        displacements: NDArray[np.float64] | None = None,
        same_layer_shape: bool = False,
        eigenvalue_tolerance: float = 1.0e-7,
    ) -> FixedShearHessian:
        """Diagonalize the non-affine Hessian at prescribed shear.

        Every projected coordinate has zero mean within each layer.  This
        removes both rigid translations and uniform relative layer
        translations, so an affine z-shear cannot be mistaken for a
        prism-breaking mode.  Set ``same_layer_shape=True`` to analyze only
        modes whose basal and apical non-affine displacements are identical.

        A scalar ``shear`` denotes a global-x shear component.  Pass the
        two-component value returned by :meth:`edge_aligned_shear` for the
        edge-aligned convention used in the public Figure 4 notebook.
        """
        if not np.isfinite(eigenvalue_tolerance) or eigenvalue_tolerance < 0.0:
            raise ValueError("eigenvalue_tolerance must be finite and non-negative.")
        shear_values = np.asarray(
            [float(shear), 0.0] if np.ndim(shear) == 0 else shear, dtype=np.float64
        )
        if shear_values.shape != (2,) or not np.all(np.isfinite(shear_values)):
            raise ValueError(
                "shear must be a finite scalar q_x or a finite two-component vector."
            )
        displacement_values = (
            np.zeros((2, self.vertex_count, 2), dtype=np.float64)
            if displacements is None
            else np.asarray(displacements, dtype=np.float64)
        )
        if displacement_values.shape != (2, self.vertex_count, 2):
            raise ValueError("displacements has the wrong shape.")
        if not np.all(np.isfinite(displacement_values)):
            raise ValueError("displacements must be finite.")

        basis = (
            self._shared_layer_basis() if same_layer_shape else self._nonaffine_basis()
        )
        flat = jnp.asarray(
            displacement_values.reshape(-1), dtype=self.reference_xy.dtype
        )
        shear_jax = jnp.asarray(shear_values, dtype=self.reference_xy.dtype)

        def flat_energy(flat_displacement: Array) -> Array:
            current = flat_displacement.reshape((2, self.vertex_count, 2))
            return self.energy(current, shear_jax, parameters)

        energy_value, gradient = jax.value_and_grad(flat_energy)(flat)
        hessian = np.asarray(jax.hessian(flat_energy)(flat), dtype=np.float64)
        hessian = 0.5 * (hessian + hessian.T)
        projected = basis.T @ hessian @ basis
        projected = 0.5 * (projected + projected.T)
        eigenvalues, eigenvectors_reduced = np.linalg.eigh(projected)
        eigenvectors = basis @ eigenvectors_reduced
        projected_gradient = basis.T @ np.asarray(gradient, dtype=np.float64)

        negative = eigenvalues < -eigenvalue_tolerance
        positive = eigenvalues > eigenvalue_tolerance
        zero = ~(negative | positive)
        minimum_mode = eigenvectors[:, 0]
        return FixedShearHessian(
            energy=float(energy_value),
            gradient_max_abs=float(np.max(np.abs(projected_gradient))),
            eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            eigenvectors=np.asarray(eigenvectors, dtype=np.float64),
            projected_hessian=np.asarray(projected, dtype=np.float64),
            negative_count=int(np.count_nonzero(negative)),
            zero_count=int(np.count_nonzero(zero)),
            positive_count=int(np.count_nonzero(positive)),
            minimum_eigenvalue=float(eigenvalues[0]),
            most_unstable_mode=np.asarray(
                minimum_mode.reshape((2, self.vertex_count, 2)), dtype=np.float64
            ),
        )
