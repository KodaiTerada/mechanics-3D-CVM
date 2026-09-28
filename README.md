# Epithelial cell vertex model

Simulation code and figure notebooks for *Epithelial mechanics in a three-dimensional cell vertex model with local junctional constraints*.

## Install and run

Use Python 3.11 or later in a dedicated environment. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks,test]'
python -m ipykernel install --sys-prefix --name epithelial-cvm3d
python -m jupyterlab notebooks
```

Select the `epithelial-cvm3d` kernel and run a notebook from top to bottom.
If the public dependencies are already installed in your kernel, you can also
open the notebooks directly: the first cell locates the bundled `cvm3d` from
this directory or any of its subdirectories, including `notebooks`.
Keep the directory structure intact. Restart the kernel if a different copy of
`cvm3d` was imported previously.
Notebooks display figures.
Double precision is enabled before calculations. `requirements-tested.txt` records
the numerical and plotting versions used for validation (Python 3.13); these can
be selected with `python -m pip install -r requirements-tested.txt`.

| Notebook | Content |
| --- | --- |
| `fig2.ipynb` | Linear and finite-strain Poisson ratios |
| `fig3.ipynb` | Regular-prism equilibria and affine elastic moduli |
| `fig4.ipynb` | Affine out-of-plane shear stability and fixed-length energy landscapes |
| `fig5.ipynb` | Bulk/shear dynamic moduli, mode coupling, relaxation spectrum; Figure S5 complex in-plane Poisson ratio |
| `figS1.ipynb` | Elastic-ratio bifurcations |
| `figS2.ipynb` | Individual dynamic responses and relaxed tilings |
| `figS3.ipynb` | Cell-number dependence: 20 and 80 cells |
| `figS4.ipynb` | Volume penalties: 1000 and 100000 |

The full dynamic notebooks use 30 realizations (seeds 100–129) and can take hours.
They stop on failed equilibration or unstable Hessians. Reducing `seeds` is useful
for a trial run (at least two for confidence intervals), but changes the ensemble.
Figures 5 and S2 use a 5 × 8 reference; S4 uses 10 × 4, as specified in its notebook.
To display S2 after running Fig. 5 in the same kernel, call `fig_s2 = plot_individuals(result)`.
Figure S5 is included in the final plotting cell of `fig5.ipynb` and reuses the Figure 5 results without additional simulation.

## Model

`cvm3d` supplies periodic mesh construction, vertex relaxation with T1 exchanges,
affine energies, and fixed-topology Hessian response. It implements prismatic
cells with identical apical/basal footprints and common height for dynamic
response, and independent planar layers for transverse-shear calculations;
it is not an unrestricted polyhedral tissue simulator.

The energy is the sum of basal area, half the lateral area, apical perimeter,
and `(kappa/3) Tr(S^-1)` at every cell corner on both layers. There is no
vertex-sharing factor or eigenvalue clipping. All notebooks use dimensionless
units, target reference volume 1, and `kappa=0.1`.
In-plane moduli are normalized by basal area; transverse shear by cell volume.
Static bulk/pure-shear derivatives use axial strain and the factor `1/(4 A)`;
dynamic bulk uses logarithmic area strain and dynamic shear uses engineering
simple shear, with normalization `1/A`.

For disordered response, the box and total volume stay fixed. Cells first relax
without a volume penalty; their converged volumes become fixed targets for
`(K_V/2) sum((V_i/V_i_target - 1)^2)`. Finite `K_V` is a soft constraint.
The force tolerance is `1e-5`; rigid translations are removed before testing
Hessian stability. Spectra use 50 logarithmic bins and bands use pointwise
95% bootstrap confidence intervals.

Run the mathematical and numerical checks with `python -m pytest -q`.
