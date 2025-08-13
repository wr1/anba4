import pytest
import numpy as np
from dolfin import *
from anba4 import *

def compute_rotation(singular):
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["quadrature_degree"] = 2
    e_xx = 9.8e9
    e_yy = 9.8e9
    e_zz = 1.42e11
    g_xy = 4.8e9
    g_xz = 6.0e9
    g_yz = 6.0e9
    nu_xy = 0.34
    nu_zx = 0.3
    nu_zy = 0.3
    matMechanicProp = np.zeros((3, 3))
    matMechanicProp[0, 0] = e_xx
    matMechanicProp[0, 1] = e_yy
    matMechanicProp[0, 2] = e_zz
    matMechanicProp[1, 0] = g_yz
    matMechanicProp[1, 1] = g_xz
    matMechanicProp[1, 2] = g_xy
    matMechanicProp[2, 0] = nu_zy
    matMechanicProp[2, 1] = nu_zx
    matMechanicProp[2, 2] = nu_xy
    sectionWidth = 3.0023e-2
    sectionHeight = 1.9215e-3
    nPly = 16
    mesh = RectangleMesh(Point(0.0, 0.0), Point(sectionWidth, sectionHeight), 30, 32, "crossed")
    ALE.move(mesh, Constant([-sectionWidth / 2.0, -sectionHeight / 2.0]))
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    tol = 1e-14
    subdomain_0_p20 = CompiledSubDomain("x[1] >= -8.0*thickness - tol && x[1] <= -7.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_1_m70 = CompiledSubDomain("x[1] >= -7.0*thickness - tol && x[1] <= -6.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_2_m70 = CompiledSubDomain("x[1] >= -6.0*thickness - tol && x[1] <= -5.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_3_p20 = CompiledSubDomain("x[1] >= -5.0*thickness - tol && x[1] <= -4.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_4_p20 = CompiledSubDomain("x[1] >= -4.0*thickness - tol && x[1] <= -3.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_5_m70 = CompiledSubDomain("x[1] >= -3.0*thickness - tol && x[1] <= -2.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_6_m70 = CompiledSubDomain("x[1] >= -2.0*thickness - tol && x[1] <= -1.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_7_p20 = CompiledSubDomain("x[1] >= -1.0*thickness - tol && x[1] <= -0.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_8_m20 = CompiledSubDomain("x[1] >= 0.0*thickness - tol && x[1] <= 1.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_9_p70 = CompiledSubDomain("x[1] >= 1.0*thickness - tol && x[1] <= 2.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_10_p70 = CompiledSubDomain("x[1] >= 2.0*thickness - tol && x[1] <= 3.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_11_m20 = CompiledSubDomain("x[1] >= 3.0*thickness - tol && x[1] <= 4.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_12_m20 = CompiledSubDomain("x[1] >= 4.0*thickness - tol && x[1] <= 5.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_13_p70 = CompiledSubDomain("x[1] >= 5.0*thickness - tol && x[1] <= 6.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_14_p70 = CompiledSubDomain("x[1] >= 6.0*thickness - tol && x[1] <= 7.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    subdomain_15_m20 = CompiledSubDomain("x[1] >= 7.0*thickness - tol && x[1] <= 8.0*thickness + tol", thickness=sectionHeight / nPly, tol=tol)
    rotation_angle = 0.0
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(rotation_angle)
    subdomain_0_p20.mark(fiber_orientations, 20.0)
    subdomain_1_m70.mark(fiber_orientations, -70.0)
    subdomain_2_m70.mark(fiber_orientations, -70.0)
    subdomain_3_p20.mark(fiber_orientations, 20.0)
    subdomain_4_p20.mark(fiber_orientations, 20.0)
    subdomain_5_m70.mark(fiber_orientations, -70.0)
    subdomain_6_m70.mark(fiber_orientations, -70.0)
    subdomain_7_p20.mark(fiber_orientations, 20.0)
    subdomain_8_m20.mark(fiber_orientations, -20.0)
    subdomain_9_p70.mark(fiber_orientations, 70.0)
    subdomain_10_p70.mark(fiber_orientations, 70.0)
    subdomain_11_m20.mark(fiber_orientations, -20.0)
    subdomain_12_m20.mark(fiber_orientations, -20.0)
    subdomain_13_p70.mark(fiber_orientations, 70.0)
    subdomain_14_p70.mark(fiber_orientations, 70.0)
    subdomain_15_m20.mark(fiber_orientations, -20.0)
    rotate = Expression(("x[0] * (cos(rotation_angle)-1.0) - x[1] * sin(rotation_angle)", "x[0] * sin(rotation_angle) + x[1] * (cos(rotation_angle)-1.0)"), rotation_angle=rotation_angle * np.pi / 180.0, degree=1)
    ALE.move(mesh, rotate)
    mat1 = material.OrthotropicMaterial(matMechanicProp)
    matLibrary = [mat1]
    anbax_data = initialize_anba_model(
        mesh,
        1,
        matLibrary,
        materials,
        plane_orientations,
        fiber_orientations,
        singular=singular,
        scaling_constraint=1.0e9,
    )
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    mass = compute_inertia(anbax_data)
    return stiff.getValues(range(6), range(6)), mass.getValues(range(6), range(6))

def test_rotation_regular_vs_singular():
    stiff_reg, mass_reg = compute_rotation(False)
    stiff_sing, mass_sing = compute_rotation(True)
    np.testing.assert_allclose(stiff_reg, stiff_sing, atol=1e-6)
    np.testing.assert_allclose(mass_reg, mass_sing, atol=1e-6)

