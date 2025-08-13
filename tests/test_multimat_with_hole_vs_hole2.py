import pytest
import numpy as np
from dolfin import *
from anba4 import *
import mshr

def compute_multimat_with_hole():
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
        scaling_constraint=1,
    )
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    return stiff.getValues(range(6), range(6))

def compute_multimat_with_hole2():
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["quadrature_degree"] = 2
    matMechanicProp1 = [80000, 0.3]
    matMechanicProp2 = [80000 * 0.5, 0.3]
    matMechanicProp3 = [80000 * 0.001, 0.3]
    Square = mshr.Rectangle(Point(-10.0, -10.0), Point(10.0, 10.0))
    Rectangle = mshr.Rectangle(Point(-2.0, -6.0), Point(2.0, 6.0))
    Domain = Square - Rectangle
    mesh = mshr.generate_mesh(Domain, 64)
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    tol = 1e-14
    lower_portion = CompiledSubDomain("x[1] <= 0 + tol", tol=tol)
    rotation_angle = 0.0
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(rotation_angle)
    lower_portion.mark(materials, 1)
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
        scaling_constraint=1,
    )
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    return stiff.getValues(range(6), range(6))

def test_multimat_with_hole_vs_hole2():
    stiff1 = compute_multimat_with_hole()
    stiff2 = compute_multimat_with_hole2()
    np.testing.assert_allclose(stiff1, stiff2, rtol=1e-3, atol=1e-3)

