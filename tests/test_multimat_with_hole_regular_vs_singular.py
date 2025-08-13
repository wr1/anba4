import pytest
import numpy as np
from dolfin import *
from anba4 import *

def compute_multimat_with_hole(singular):
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["quadrature_degree"] = 2
    matMechanicProp1 = [80000, 0.3]
    matMechanicProp2 = [80000 * 0.5, 0.3]
    matMechanicProp3 = [80000 * 0.00001, 0.3]
    sectionWidth = 20
    sectionHeight = 20
    mesh = RectangleMesh(Point(0.0, 0.0), Point(sectionWidth, sectionHeight), 50, 50, "crossed")
    ALE.move(mesh, Constant([-sectionWidth / 2.0, -sectionHeight / 2.0]))
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    tol = 1e-14
    lower_portion = CompiledSubDomain("x[1] <= 0 + tol", tol=tol)
    hole = CompiledSubDomain("(x[1] >= -6 + tol && x[1] <= 6. + tol)&&(x[0] >= -2 + tol && x[0] <= 2. + tol)", tol=tol)
    rotation_angle = 0.0
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(rotation_angle)
    lower_portion.mark(materials, 1)
    hole.mark(materials, 2)
    mat1 = material.IsotropicMaterial(matMechanicProp1)
    mat2 = material.IsotropicMaterial(matMechanicProp2)
    mat3 = material.IsotropicMaterial(matMechanicProp3)
    matLibrary = [mat1, mat2, mat3]
    anbax_data = initialize_anba_model(
        mesh,
        1,
        matLibrary,
        materials,
        plane_orientations,
        fiber_orientations,
        singular=singular,
        scaling_constraint=1,
    )
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    mass = compute_inertia(anbax_data)
    return stiff.getValues(range(6), range(6)), mass.getValues(range(6), range(6))

def test_multimat_with_hole_regular_vs_singular():
    stiff_reg, mass_reg = compute_multimat_with_hole(False)
    stiff_sing, mass_sing = compute_multimat_with_hole(True)
    np.testing.assert_allclose(stiff_reg, stiff_sing, atol=1e-5)
    np.testing.assert_allclose(mass_reg, mass_sing, atol=1e-5)

