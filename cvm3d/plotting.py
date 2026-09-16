"""Display figure observables without writing files."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from cvm3d.geometry import face_vertices_unfolded


def mean_band(values, samples=5000, seed=5009):
    """Pointwise percentile bootstrap CI, resampling whole realizations."""
    values = np.asarray(values)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError(
            "Confidence intervals require at least two finite realizations."
        )
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 100):
        indices = rng.integers(
            len(values), size=(min(100, samples - start), len(values))
        )
        means.append(values[indices].mean(axis=1))
    lower, upper = np.percentile(np.concatenate(means), [2.5, 97.5], axis=0)
    return values.mean(axis=0), lower, upper


def modes(result, response, mesh):
    active = response.hessian_eigenvalues > result["protocol"].eigenvalue_tolerance
    h, xi = response.hessian_eigenvalues[active], response.strain_couplings[active]
    weights = xi**2 / (mesh.width * mesh.height * h)
    if response.relaxed_modulus is None:
        raise ValueError("Strain couples to a zero Hessian mode.")
    np.testing.assert_allclose(
        weights.sum(),
        response.affine_modulus - response.relaxed_modulus,
        rtol=1e-8,
        atol=1e-10,
    )
    return dict(
        h=h / result["h0"],
        xi=np.abs(xi),
        log_tau=np.log(result["h0"] / h),
        weights=weights,
    )


def plot_dynamic(results, bins=50, shoulder=(0.005, 0.025)):
    """Six panels per condition; shared spectral bins across displayed cases."""
    modal = [
        [modes(r, r["regular_shear"], r["regular"])]
        + [modes(r, s, m) for s, m in zip(r["shear"], r["meshes"], strict=True)]
        for r in results
    ]
    all_times = np.concatenate([m["log_tau"] for group in modal for m in group])
    edges = np.linspace(all_times.min(), all_times.max(), bins + 1)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)

    def spectrum(m):
        histogram = np.histogram(m["log_tau"], edges, weights=m["weights"])[0] / widths
        np.testing.assert_allclose(
            np.sum(histogram * widths), m["weights"].sum(), rtol=1e-10, atol=1e-12
        )
        return histogram

    fig, axes = plt.subplots(
        2 * len(results), 3, figsize=(12, 6 * len(results)), layout="constrained"
    )
    for row, (r, group) in enumerate(zip(results, modal, strict=True)):
        a, b, c, d, e, f = axes[2 * row : 2 * row + 2].ravel()
        frequency = r["frequencies"] / r["omega0"]
        for ax, key, attribute, label in [
            (a, "bulk", "storage_modulus", r"$K'$"),
            (b, "shear", "storage_modulus", r"$G'$"),
            (d, "bulk", "loss_modulus", r"$K''$"),
            (e, "shear", "loss_modulus", r"$G''$"),
        ]:
            mean, lo, hi = mean_band([getattr(s, attribute) for s in r[key]])
            color = "tab:blue" if attribute == "storage_modulus" else "tab:orange"
            ax.fill_between(
                frequency,
                np.maximum(lo, 1e-16),
                np.maximum(hi, 1e-16),
                color=color,
                alpha=0.2,
                label="95% CI",
            )
            ax.loglog(
                frequency, np.maximum(mean, 1e-16), color=color, label="disordered mean"
            )
            if ax is not d:
                ax.loglog(
                    frequency,
                    np.maximum(getattr(r["regular_" + key], attribute), 1e-16),
                    color="black",
                    label="regular",
                )
            ax.set(xlabel=r"$\omega/\omega_0$", ylabel=label)
        p, l = r["parameters"], r["l0"]
        K = (
            p.gamma_b / 2
            + 2 * p.cell_volume * p.gamma_c / (9 * l**3)
            + 32 * np.sqrt(3) * p.kappa / (9 * l**4)
            + 18 * np.sqrt(3) * p.kappa * l**2 / p.cell_volume**2
        )
        G = (
            p.cell_volume * p.gamma_c / (6 * l**3)
            + np.sqrt(3) * p.sigma / (2 * l)
            + 64 * np.sqrt(3) * p.kappa / (27 * l**4)
        )
        for ax, val in ((a, K), (b, G)):
            ax.axhline(val, color="tab:green", ls=":", label="affine theory")
        for m in group[1:]:
            c.scatter(
                m["h"], np.maximum(m["xi"], 1e-5), s=2, alpha=0.18, color="tab:blue"
            )
        c.scatter(
            group[0]["h"],
            np.maximum(group[0]["xi"], 1e-5),
            s=7,
            marker="x",
            color="black",
        )
        c.set(xscale="log", yscale="log", xlabel=r"$h_j/h_0$", ylabel=r"$|\xi_{G,j}|$")
        mean, lo, hi = mean_band([spectrum(m) for m in group[1:]])
        f.fill_between(centers, np.maximum(lo, 0), hi, color="tab:blue", alpha=0.2)
        f.plot(centers, mean, color="tab:blue")
        f.plot(centers, spectrum(group[0]), color="black")
        f.set(xlabel=r"$\ln(\tau/\tau_0)$", ylabel=r"$S_G$", ylim=(0, None))
        for ax in (c, e):
            ax.axvspan(*shoulder, color="tab:purple", alpha=0.12)
        f.axvspan(
            -np.log(shoulder[1]), -np.log(shoulder[0]), color="tab:purple", alpha=0.12
        )
        for i, ax in enumerate((a, b, c, d, e, f)):
            ax.set_title(chr(97 + 6 * row + i), loc="left")
            ax.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=7)
        b.set_title(f"N={r['regular'].nb_faces}, K_V={r['volume_penalty']:g}")
    plt.show()
    return fig


def periodic_polygons(mesh):
    box = np.array([mesh.width, mesh.height])
    polygons = []
    for face in range(mesh.nb_faces):
        points, valid = face_vertices_unfolded(
            jnp.asarray(face),
            mesh.vertices,
            mesh.half_edges,
            mesh.faces,
            mesh.width,
            mesh.height,
            max_edges=mesh.max_edges,
        )
        points = np.asarray(points)[np.asarray(valid)]
        points = points - np.floor(points.mean(axis=0) / box) * box
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                polygons.append(points + box * np.array([ix, iy]))
    return polygons


def plot_individuals(result, columns=6):
    count = len(result["meshes"])
    rows = int(np.ceil(count / columns))
    fig = plt.figure(figsize=(12, 3 + 2 * rows), layout="constrained")
    layout = fig.add_gridspec(2, 1, height_ratios=[3, 2 * rows])
    top = layout[0].subgridspec(1, 2)
    bottom = layout[1].subgridspec(rows, columns)
    colors = plt.colormaps["turbo"](np.linspace(0.03, 0.97, count))
    frequency = result["frequencies"] / result["omega0"]
    for col, (key, label) in enumerate((("bulk", "K"), ("shear", "G"))):
        ax = fig.add_subplot(top[col])
        for response, color in zip(result[key], colors, strict=True):
            ax.loglog(
                frequency, np.maximum(response.storage_modulus, 1e-16), color=color
            )
            ax.loglog(
                frequency, np.maximum(response.loss_modulus, 1e-16), "--", color=color
            )
        ax.set(
            xlabel=r"$\omega/\omega_0$",
            ylabel=label,
            title="Storage (solid), loss (dashed)",
        )
    for i, (mesh, seed, color) in enumerate(
        zip(result["meshes"], result["seeds"], colors, strict=True)
    ):
        ax = fig.add_subplot(bottom[i // columns, i % columns])
        ax.add_collection(
            PolyCollection(
                periodic_polygons(mesh),
                facecolors=[(*color[:3], 0.12)],
                edgecolors=".25",
                linewidths=0.4,
            )
        )
        ax.set(
            xlim=(0, mesh.width),
            ylim=(0, mesh.height),
            aspect="equal",
            xticks=[],
            yticks=[],
            title=f"seed {seed}",
        )
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
    plt.show()
    return fig
