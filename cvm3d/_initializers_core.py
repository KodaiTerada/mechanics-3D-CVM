"""Mesh initializers for periodic prismatic CVMs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cvm3d.mesh import PrismaticMesh3D


def triangular_lattice_seeds(
    nx: int,
    ny: int,
    side_length: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Return seeds and box size for a periodic regular hexagonal tiling.

    The Voronoi cells of the returned triangular seed lattice are regular
    hexagons with side length ``side_length``. ``ny`` must be even so that the
    staggered rows fit in a rectangular periodic box.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers.")
    if ny % 2 != 0:
        raise ValueError(
            "ny must be even for a rectangular periodic triangular lattice."
        )
    if side_length <= 0.0:
        raise ValueError("side_length must be positive.")

    nearest_neighbor_distance = np.sqrt(3.0) * float(side_length)
    row_spacing = 1.5 * float(side_length)
    width = int(nx) * nearest_neighbor_distance
    height = int(ny) * row_spacing
    seeds = np.asarray(
        [
            [
                ((i + 0.25 + 0.5 * (j % 2)) * nearest_neighbor_distance) % width,
                (j + 0.5) * row_spacing,
            ]
            for j in range(ny)
            for i in range(nx)
        ],
        dtype=np.float64,
    )
    return seeds, float(width), float(height)


def make_regular_hexagonal_mesh(
    nx: int,
    ny: int,
    side_length: float,
    cell_volume: float,
    *,
    gamma_b: float | None = None,
    gamma_c: float | None = None,
    sigma: float | None = None,
    use_local_cell_volumes: bool = False,
    max_edges: int = 20,
) -> PrismaticMesh3D:
    """Return a periodic prismatic mesh of uniform regular hexagonal cells.

    When ``gamma_b``, ``gamma_c``, and ``sigma`` are supplied, uniform face
    parameter columns are populated in that order. They must be supplied
    together. If ``use_local_cell_volumes`` is true, a fourth cell-wise volume
    column is added so that the energy uses ``h_i = V_i / A_i``.
    """
    from cvm3d.mesh import PrismaticMesh3D

    if cell_volume <= 0.0:
        raise ValueError("cell_volume must be positive.")
    if max_edges < 3:
        raise ValueError("max_edges must be at least 3.")
    supplied_parameters = (gamma_b is not None, gamma_c is not None, sigma is not None)
    if any(supplied_parameters) and not all(supplied_parameters):
        raise ValueError("gamma_b, gamma_c, and sigma must be supplied together.")
    if use_local_cell_volumes and not all(supplied_parameters):
        raise ValueError(
            "gamma_b, gamma_c, and sigma are required when use_local_cell_volumes=True "
            "because cell volume is face parameter column 3."
        )

    seeds, width, height = triangular_lattice_seeds(nx, ny, side_length)
    vertices, half_edges, faces = periodic_voronoi_tables(
        seeds,
        width,
        height,
        allowed_face_degrees={6},
        max_edges=max_edges,
    )
    mesh = PrismaticMesh3D(
        vertices=jnp.asarray(vertices),
        half_edges=jnp.asarray(half_edges, dtype=jnp.int32),
        faces=jnp.asarray(faces, dtype=jnp.int32),
        width=width,
        height=height,
        total_volume=float(nx * ny * cell_volume),
        max_edges=max_edges,
    )
    if all(supplied_parameters):
        parameter_columns = [
            jnp.full(mesh.nb_faces, gamma_b),
            jnp.full(mesh.nb_faces, gamma_c),
            jnp.full(mesh.nb_faces, sigma),
        ]
        if use_local_cell_volumes:
            parameter_columns.append(jnp.full(mesh.nb_faces, cell_volume))
        mesh.faces_params = jnp.column_stack(parameter_columns)
    return mesh


def perturb_mesh_vertices(
    mesh: PrismaticMesh3D,
    amplitude: float,
    random_key: int | Array = 0,
    *,
    remove_mean: bool = True,
) -> PrismaticMesh3D:
    """Return a copy of ``mesh`` with Gaussian vertex-position perturbations.

    ``amplitude`` is the Gaussian standard deviation in coordinate units. A
    scalar integer is converted to a JAX PRNG key; an existing JAX key can also
    be supplied. By default the mean displacement is removed to avoid adding a
    rigid translation. Topology, box dimensions, and parameter tables are
    preserved, and the perturbed coordinates are wrapped into the periodic box.
    """
    if amplitude < 0.0:
        raise ValueError("amplitude must be non-negative.")
    key = (
        jax.random.PRNGKey(int(random_key))
        if isinstance(random_key, (int, np.integer))
        else jnp.asarray(random_key)
    )
    perturbed = mesh.copy()
    noise = jax.random.normal(
        key, perturbed.vertices.shape, dtype=perturbed.vertices.dtype
    )
    if remove_mean:
        noise = noise - jnp.mean(noise, axis=0, keepdims=True)
    perturbed.vertices = perturbed.vertices + float(amplitude) * noise
    perturbed.update_boundary_conditions()
    return perturbed


def periodic_rms_displacement(
    vertices_a: Array,
    vertices_b: Array,
    width: float,
    height: float,
) -> Array:
    """Return RMS in-plane vertex displacement using minimum-image distances."""
    first = jnp.asarray(vertices_a)
    second = jnp.asarray(vertices_b)
    if first.shape != second.shape:
        raise ValueError("vertices_a and vertices_b must have the same shape.")
    if first.ndim != 2 or first.shape[1] < 2:
        raise ValueError(
            "vertex arrays must have shape (n_vertices, n_coordinates) with at least x and y."
        )
    if first.shape[0] == 0:
        raise ValueError("vertex arrays must not be empty.")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")
    box = jnp.asarray([width, height], dtype=jnp.result_type(first, second))
    delta = first[:, :2] - second[:, :2]
    delta = delta - jnp.round(delta / box) * box
    return jnp.sqrt(jnp.mean(jnp.sum(delta**2, axis=1)))


def face_vertex_counts(
    half_edges: NDArray[np.integer],
    faces: NDArray[np.integer],
    max_edges: int = 20,
) -> NDArray[np.int32]:
    """Return the number of vertices/edges around each face."""
    counts: list[int] = []
    for start_he in np.asarray(faces, dtype=np.int32):
        he = int(start_he)
        count = 0
        for _ in range(max_edges):
            count += 1
            he = int(half_edges[he, 1])
            if he == int(start_he):
                break
        else:
            msg = f"Face starting at half-edge {int(start_he)} did not close within max_edges={max_edges}."
            raise ValueError(msg)
        counts.append(count)
    return np.asarray(counts, dtype=np.int32)


def _normalize_allowed_face_degrees(
    allowed_face_degrees: Iterable[int] | None,
) -> set[int] | None:
    if allowed_face_degrees is None:
        return None
    allowed = {int(degree) for degree in allowed_face_degrees}
    if not allowed:
        msg = "allowed_face_degrees must be None or a non-empty iterable of integers."
        raise ValueError(msg)
    if any(degree < 3 for degree in allowed):
        msg = "allowed_face_degrees must contain polygon vertex counts >= 3."
        raise ValueError(msg)
    return allowed


def _validate_face_degrees(
    half_edges: NDArray[np.integer],
    faces: NDArray[np.integer],
    allowed_face_degrees: set[int] | None,
    max_edges: int = 20,
) -> NDArray[np.int32]:
    counts = face_vertex_counts(half_edges, faces, max_edges=max_edges)
    if allowed_face_degrees is not None and not np.all(
        np.isin(counts, list(allowed_face_degrees))
    ):
        unique, frequencies = np.unique(counts, return_counts=True)
        distribution = dict(
            zip(
                unique.astype(int).tolist(),
                frequencies.astype(int).tolist(),
                strict=False,
            )
        )
        msg = f"Voronoi mesh contains disallowed face degrees: {distribution}; allowed={sorted(allowed_face_degrees)}."
        raise ValueError(msg)
    return counts


def default_rsa_min_distance(nb_seeds: int, width: float, height: float) -> float:
    """Return a conservative automatic RSA exclusion distance."""
    if nb_seeds <= 0:
        msg = "nb_seeds must be a positive integer."
        raise ValueError(msg)
    if width <= 0.0 or height <= 0.0:
        msg = "width and height must be positive."
        raise ValueError(msg)
    triangular_lattice_spacing = np.sqrt(
        2.0 * float(width) * float(height) / (np.sqrt(3.0) * int(nb_seeds))
    )
    return float(0.55 * triangular_lattice_spacing)


def _periodic_distances(
    candidate: NDArray[np.float64],
    seeds: NDArray[np.float64],
    width: float,
    height: float,
) -> NDArray[np.float64]:
    delta = np.abs(seeds - candidate)
    delta[:, 0] = np.minimum(delta[:, 0], width - delta[:, 0])
    delta[:, 1] = np.minimum(delta[:, 1], height - delta[:, 1])
    return np.sqrt(np.sum(delta**2, axis=1))


def _random_sequential_addition_seeds(
    rng: np.random.Generator,
    nb_seeds: int,
    width: float,
    height: float,
    min_distance: float,
    max_candidate_attempts: int,
) -> NDArray[np.float64]:
    if nb_seeds <= 0:
        msg = "nb_seeds must be a positive integer."
        raise ValueError(msg)
    if width <= 0.0 or height <= 0.0:
        msg = "width and height must be positive."
        raise ValueError(msg)
    if min_distance < 0.0:
        msg = "rsa_min_distance must be non-negative."
        raise ValueError(msg)
    if max_candidate_attempts <= 0:
        msg = "rsa_max_candidate_attempts must be a positive integer."
        raise ValueError(msg)

    seeds = np.empty((int(nb_seeds), 2), dtype=np.float64)
    accepted = 0
    attempts_since_accept = 0
    box = np.array([width, height], dtype=np.float64)
    while accepted < int(nb_seeds):
        if attempts_since_accept >= int(max_candidate_attempts):
            msg = (
                f"Could not place seed {accepted + 1}/{int(nb_seeds)} with RSA min_distance={min_distance} "
                f"after {int(max_candidate_attempts)} candidate attempts."
            )
            raise ValueError(msg)
        candidate = rng.random(2) * box
        if accepted == 0 or np.all(
            _periodic_distances(candidate, seeds[:accepted], width, height)
            >= min_distance
        ):
            seeds[accepted] = candidate
            accepted += 1
            attempts_since_accept = 0
        else:
            attempts_since_accept += 1
    return seeds


def random_sequential_addition_seeds(
    nb_seeds: int,
    width: float,
    height: float,
    random_key: int,
    min_distance: float | None = None,
    max_candidate_attempts: int = 10000,
) -> NDArray[np.float64]:
    """Return seed points generated by periodic random sequential addition."""
    distance = (
        default_rsa_min_distance(nb_seeds, width, height)
        if min_distance is None
        else float(min_distance)
    )
    rng = np.random.default_rng(random_key)
    return _random_sequential_addition_seeds(
        rng,
        nb_seeds,
        width,
        height,
        distance,
        max_candidate_attempts,
    )


def _periodic_seed_tiles(
    seeds: NDArray[np.float64],
    width: float,
    height: float,
) -> NDArray[np.float64]:
    offsets = np.asarray(
        [
            [0.0, 0.0],
            [-width, +height],
            [0.0, +height],
            [+width, +height],
            [-width, 0.0],
            [+width, 0.0],
            [-width, -height],
            [0.0, -height],
            [+width, -height],
        ],
        dtype=np.float64,
    )
    return np.concatenate([seeds + offset for offset in offsets], axis=0)


def _polygon_centroid(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    targets = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * targets[:, 1] - vertices[:, 1] * targets[:, 0]
    twice_area = np.sum(cross)
    if abs(twice_area) <= np.finfo(np.float64).eps:
        return np.mean(vertices, axis=0)
    return np.asarray(
        [
            np.sum((vertices[:, 0] + targets[:, 0]) * cross),
            np.sum((vertices[:, 1] + targets[:, 1]) * cross),
        ]
    ) / (3.0 * twice_area)


def periodic_lloyd_relaxation(
    seeds: NDArray[np.float64],
    width: float,
    height: float,
    iterations: int = 10,
    relaxation: float = 1.0,
) -> NDArray[np.float64]:
    """Move periodic Voronoi seeds toward their cell centroids.

    ``relaxation=1`` performs standard Lloyd iterations. Smaller positive
    values under-relax each update. Returned seeds are wrapped into the
    rectangular periodic box.
    """
    try:
        from scipy.spatial import Voronoi
    except ModuleNotFoundError as exc:
        msg = "scipy is required for periodic Lloyd relaxation."
        raise ModuleNotFoundError(msg) from exc

    current = np.asarray(seeds, dtype=np.float64)
    if current.ndim != 2 or current.shape[1] != 2 or current.shape[0] < 1:
        raise ValueError("seeds must have shape (n_seeds, 2) with at least one seed.")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")
    if int(iterations) < 0:
        raise ValueError("iterations must be non-negative.")
    if not 0.0 < float(relaxation) <= 1.0:
        raise ValueError("relaxation must be in the interval (0, 1].")

    box = np.asarray([width, height], dtype=np.float64)
    current = np.mod(current, box)
    for _ in range(int(iterations)):
        voronoi = Voronoi(_periodic_seed_tiles(current, width, height))
        centroids = np.empty_like(current)
        for seed_id in range(current.shape[0]):
            region = voronoi.regions[voronoi.point_region[seed_id]]
            if not region or any(vertex_id < 0 for vertex_id in region):
                raise RuntimeError(
                    "Periodic Voronoi cell is unexpectedly unbounded during Lloyd relaxation."
                )
            centroids[seed_id] = _polygon_centroid(
                voronoi.vertices[np.asarray(region, dtype=np.int32)]
            )
        displacement = centroids - current
        displacement -= np.round(displacement / box) * box
        current = np.mod(current + float(relaxation) * displacement, box)
    return current


def periodic_voronoi_tables(
    seeds: NDArray[np.float64],
    width: float,
    height: float,
    allowed_face_degrees: Iterable[int] | None = None,
    max_edges: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Return VertAX-style periodic DCEL tables from in-plane seed points."""
    allowed = _normalize_allowed_face_degrees(allowed_face_degrees)
    (
        periodic_vertices_idx,
        periodic_vertices_pos,
        periodic_edges,
        offsets,
        periodic_faces,
    ) = _make_periodic(np.asarray(seeds, dtype=np.float64), width, height)
    vertices, half_edges, faces = _make_half_edge_structure(
        width,
        height,
        periodic_vertices_idx,
        periodic_vertices_pos,
        periodic_edges,
        offsets,
        periodic_faces,
    )
    _validate_face_degrees(half_edges, faces, allowed, max_edges=max_edges)
    return vertices, half_edges, faces


def random_periodic_voronoi_tables(
    nb_seeds: int,
    width: float,
    height: float,
    random_key: int,
    allowed_face_degrees: Iterable[int] | None = None,
    max_attempts: int = 1000,
    max_edges: int = 20,
    rsa_min_distance: float | None = None,
    rsa_max_candidate_attempts: int = 10000,
    lloyd_iterations: int = 0,
    lloyd_relaxation: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Return periodic DCEL tables from RSA seeds, optionally Lloyd-relaxed."""
    allowed = _normalize_allowed_face_degrees(allowed_face_degrees)
    if max_attempts <= 0:
        msg = "max_attempts must be a positive integer."
        raise ValueError(msg)
    min_distance = (
        default_rsa_min_distance(nb_seeds, width, height)
        if rsa_min_distance is None
        else float(rsa_min_distance)
    )
    rng = np.random.default_rng(random_key)
    last_counts: NDArray[np.int32] | None = None
    for _ in range(int(max_attempts)):
        seeds = _random_sequential_addition_seeds(
            rng,
            nb_seeds,
            width,
            height,
            min_distance,
            rsa_max_candidate_attempts,
        )
        seeds = periodic_lloyd_relaxation(
            seeds,
            width,
            height,
            iterations=lloyd_iterations,
            relaxation=lloyd_relaxation,
        )
        vertices, half_edges, faces = periodic_voronoi_tables(seeds, width, height)
        counts = face_vertex_counts(half_edges, faces, max_edges=max_edges)
        last_counts = counts
        if allowed is None or np.all(np.isin(counts, list(allowed))):
            return vertices, half_edges, faces

    unique, frequencies = (
        np.unique(last_counts, return_counts=True)
        if last_counts is not None
        else ([], [])
    )
    distribution = dict(
        zip(
            np.asarray(unique).astype(int).tolist(),
            np.asarray(frequencies).astype(int).tolist(),
            strict=False,
        )
    )
    msg = (
        f"Could not generate a Voronoi mesh with allowed_face_degrees={sorted(allowed) if allowed else None} "
        f"after {int(max_attempts)} attempts. Last face-degree distribution: {distribution}."
    )
    raise ValueError(msg)


def _make_periodic(
    seeds: NDArray[np.float64],
    width: float,
    height: float,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    list[tuple[int, int]],
    dict[tuple[int, int], tuple[int, int]],
    list[set[int]],
]:
    try:
        from scipy.spatial import Voronoi
    except ModuleNotFoundError as exc:
        msg = "scipy is required for periodic Voronoi initialization."
        raise ModuleNotFoundError(msg) from exc

    n_cells = len(seeds)
    if n_cells < 20:
        warnings.warn(
            "[n_cells < 20] initial condition may not work as expected.",
            stacklevel=2,
        )

    padded_seeds = _periodic_seed_tiles(seeds, width, height)
    voronoi = Voronoi(padded_seeds)
    vertices = voronoi.vertices
    edges = voronoi.ridge_vertices
    faces = voronoi.regions

    inside = (
        (vertices[:, 0] >= 0.0)
        & (vertices[:, 0] <= width)
        & (vertices[:, 1] >= 0.0)
        & (vertices[:, 1] <= height)
    )
    periodic_vertex_idx = np.arange(len(vertices), dtype=np.int32)[inside]
    periodic_vertex_pos = vertices[inside]

    inside_set = set(int(i) for i in periodic_vertex_idx)
    edges_inside: list[tuple[int, int]] = []
    edges_outside: list[tuple[int, int]] = []
    offsets_inside: dict[tuple[int, int], tuple[int, int]] = {}
    offsets_outside: dict[tuple[int, int], tuple[int, int]] = {}
    visited: set[tuple[int, int]] = set()

    for ridge in edges:
        if ridge[0] < 0 or ridge[1] < 0:
            continue
        source_in = ridge[0] in inside_set
        target_in = ridge[1] in inside_set
        if source_in and target_in:
            edge = tuple(sorted((int(ridge[0]), int(ridge[1]))))
            edges_inside.append(edge)
            offsets_inside[(int(ridge[0]), int(ridge[1]))] = (0, 0)
            offsets_inside[(int(ridge[1]), int(ridge[0]))] = (0, 0)
        elif source_in:
            wrapped, offset = _wrap_vertex(vertices[ridge[1]], width, height)
            match = _find_matching_vertex(
                wrapped, periodic_vertex_idx, periodic_vertex_pos
            )
            if match is not None:
                edge = tuple(sorted((int(ridge[0]), int(match))))
                edges_outside.append(edge)
                if (int(ridge[0]), int(ridge[1])) not in visited and (
                    int(ridge[1]),
                    int(ridge[0]),
                ) not in visited:
                    offsets_outside[(int(ridge[0]), int(match))] = offset
                    offsets_outside[(int(match), int(ridge[0]))] = (
                        -offset[0],
                        -offset[1],
                    )
                    visited.add((int(ridge[0]), int(ridge[1])))
                    visited.add((int(ridge[1]), int(ridge[0])))
        elif target_in:
            wrapped, offset = _wrap_vertex(vertices[ridge[0]], width, height)
            match = _find_matching_vertex(
                wrapped, periodic_vertex_idx, periodic_vertex_pos
            )
            if match is not None:
                edge = tuple(sorted((int(match), int(ridge[1]))))
                edges_outside.append(edge)
                if (int(ridge[0]), int(ridge[1])) not in visited and (
                    int(ridge[1]),
                    int(ridge[0]),
                ) not in visited:
                    offsets_outside[(int(match), int(ridge[1]))] = (
                        -offset[0],
                        -offset[1],
                    )
                    offsets_outside[(int(ridge[1]), int(match))] = offset
                    visited.add((int(ridge[0]), int(ridge[1])))
                    visited.add((int(ridge[1]), int(ridge[0])))

    periodic_edges = list(set(edges_inside)) + list(set(edges_outside))
    offsets = offsets_inside | offsets_outside

    faces_inside_outside = []
    for face in faces:
        if not face or any(vertex_id < 0 for vertex_id in face):
            continue
        if any(vertex_id in inside_set for vertex_id in face):
            wrapped_face = []
            for vertex_id in face:
                if vertex_id in inside_set:
                    wrapped_face.append(int(vertex_id))
                else:
                    wrapped, _ = _wrap_vertex(vertices[vertex_id], width, height)
                    match = _find_matching_vertex(
                        wrapped, periodic_vertex_idx, periodic_vertex_pos
                    )
                    if match is not None:
                        wrapped_face.append(int(match))
            if wrapped_face:
                faces_inside_outside.append(tuple(sorted(set(wrapped_face))))

    return (
        periodic_vertex_idx,
        periodic_vertex_pos,
        periodic_edges,
        offsets,
        list(set(faces_inside_outside)),
    )


def _wrap_vertex(
    position: NDArray[np.float64], width: float, height: float
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    x, y = float(position[0]), float(position[1])
    offset_x = -1 if x < 0.0 else 1 if x > width else 0
    offset_y = -1 if y < 0.0 else 1 if y > height else 0
    wrapped_x = x + width if x < 0.0 else x - width if x > width else x
    wrapped_y = y + height if y < 0.0 else y - height if y > height else y
    return np.array([wrapped_x, wrapped_y]), (offset_x, offset_y)


def _find_matching_vertex(
    position: NDArray[np.float64],
    vertex_indices: NDArray[np.int32],
    vertex_positions: NDArray[np.float64],
    tolerance: float = 1e-8,
) -> int | None:
    for idx, candidate in zip(vertex_indices, vertex_positions, strict=False):
        if np.max(np.abs(candidate - position)) < tolerance:
            return int(idx)
    return None


def _make_half_edge_structure(
    width: float,
    height: float,
    periodic_vertex_idx: NDArray[np.int32],
    periodic_vertex_pos: NDArray[np.float64],
    periodic_edges: list[tuple[int, int]],
    offsets: dict[tuple[int, int], tuple[int, int]],
    periodic_faces: list[set[int]],
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    periodic_half_edges = []
    for edge in periodic_edges:
        periodic_half_edges.append(edge)
        periodic_half_edges.append((edge[1], edge[0]))

    ordered_faces = []
    for face in periodic_faces:
        face_edges = [
            (f1, f2) for f1 in face for f2 in face if (f1, f2) in periodic_edges
        ]
        if not face_edges:
            continue
        i = 0
        start_edge = face_edges[i]
        ordered_face = [start_edge]
        edge = start_edge
        visited = [edge]
        while sorted(face_edges) != sorted(visited):
            if edge[0] == start_edge[1] and edge not in visited:
                ordered_face.append(edge)
                start_edge = edge
                visited.append(edge)
            if edge[1] == start_edge[1] and edge not in visited:
                ordered = (edge[1], edge[0])
                ordered_face.append(ordered)
                start_edge = ordered
                visited.append(edge)
            i += 1
            edge = face_edges[i % len(face)]

        order = _face_order(
            ordered_face,
            periodic_vertex_idx,
            periodic_vertex_pos,
            offsets,
            width,
            height,
        )
        if order < 0:
            ordered_faces.append(ordered_face)
        elif order > 0:
            ordered_faces.append(
                [(edge[1], edge[0]) for edge in reversed(ordered_face)]
            )
        else:
            msg = f"No orientation detected for face {face}."
            raise ValueError(msg)

    vertex_index_to_local = {
        int(idx): local for local, idx in enumerate(periodic_vertex_idx)
    }
    face_table = np.zeros(len(ordered_faces), dtype=np.int32)
    for face_id, face_edges in enumerate(ordered_faces):
        face_table[face_id] = periodic_half_edges.index(face_edges[0])

    half_edge_table = np.zeros((len(periodic_half_edges), 8), dtype=np.int32)
    for half_edge_id, half_edge in enumerate(periodic_half_edges):
        for face_id, face_edges in enumerate(ordered_faces):
            if half_edge in face_edges:
                idx = face_edges.index(half_edge)
                half_edge_table[half_edge_id, 0] = periodic_half_edges.index(
                    face_edges[(idx - 1) % len(face_edges)]
                )
                half_edge_table[half_edge_id, 1] = periodic_half_edges.index(
                    face_edges[(idx + 1) % len(face_edges)]
                )
                half_edge_table[half_edge_id, 3] = vertex_index_to_local[half_edge[0]]
                half_edge_table[half_edge_id, 4] = vertex_index_to_local[half_edge[1]]
                half_edge_table[half_edge_id, 5] = face_id
                break
        half_edge_table[half_edge_id, 2] = periodic_half_edges.index(
            (half_edge[1], half_edge[0])
        )
        half_edge_table[half_edge_id, 6] = offsets[half_edge][0]
        half_edge_table[half_edge_id, 7] = offsets[half_edge][1]
    return periodic_vertex_pos, half_edge_table, face_table


def _face_order(
    ordered_face: list[tuple[int, int]],
    periodic_vertex_idx: NDArray[np.int32],
    periodic_vertex_pos: NDArray[np.float64],
    offsets: dict[tuple[int, int], tuple[int, int]],
    width: float,
    height: float,
) -> float:
    vertex_index_to_local = {
        int(idx): local for local, idx in enumerate(periodic_vertex_idx)
    }
    offset_x = 0
    offset_y = 0
    order = 0.0
    for edge in ordered_face:
        source = vertex_index_to_local[edge[0]]
        target = vertex_index_to_local[edge[1]]
        edge_offset = offsets[edge]
        prev_offset_x = offset_x
        prev_offset_y = offset_y
        offset_x += edge_offset[0]
        offset_y += edge_offset[1]
        x0 = periodic_vertex_pos[source][0] + prev_offset_x * width
        y0 = periodic_vertex_pos[source][1] + prev_offset_y * height
        x1 = periodic_vertex_pos[target][0] + offset_x * width
        y1 = periodic_vertex_pos[target][1] + offset_y * height
        order += (x1 - x0) * (y1 + y0)
    return order
