"""Minimal simulator used by the public 3D CVM figure notebooks."""

from cvm3d.initializers import make_regular_hexagonal_mesh
from cvm3d.mesh import PrismaticMesh3D
from cvm3d.rheology import complex_modulus
from cvm3d.simulate import simulate

__version__ = "0.1.0"

__all__ = [
    "PrismaticMesh3D",
    "complex_modulus",
    "make_regular_hexagonal_mesh",
    "simulate",
]
