import pytest
import numpy as np
from dolfin import *
from anba4 import *

def compute_isotropic(singular):
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["quadrature_degree"] = 2
    E = 1.0
    nu = 0.33
    matMechanicProp = [E, nu]
    mesh = UnitSquareMesh(10, 10)
    ALE.move(mesh, Constant([-0.5, -0.5]))
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(90.0)
    mat1 = material.IsotropicMaterial(matMechanicProp, 1.0)
    matLibrary = [mat1]
    anbax_data = initialize_anba_model(
        mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations, singular=singular
    )
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    mass = compute_inertia(anbax_data)
    return stiff.getValues(range(6), range(6)), mass.getValues(range(6), range(6))

def test_isotropic_regular_vs_singular():
    stiff_reg, mass_reg = compute_isotropic(False)
    stiff_sing, mass_sing = compute_isotropic(True)
    np.testing.assert_allclose(stiff_reg, stiff_sing, atol=1e-6)
    np.testing.assert_allclose(mass_reg, mass_sing, atol=1e-6)

