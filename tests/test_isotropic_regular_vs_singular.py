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
        mesh,
        2,
        matLibrary,
        materials,
        plane_orientations,
        fiber_orientations,
        singular=singular,
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


reference_stiffness = """3.1106440126718432e-01 -5.7626764670407046e-07 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.8332325847382240e-16 
-5.7626764651431983e-07 3.1106440126718449e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -2.3565142889073543e-16 
0.0000000000000000e+00 0.0000000000000000e+00 9.9999999999998768e-01 -3.9407074330915428e-16 7.1891020955786236e-17 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -4.2103095762601342e-16 8.3333333333332371e-02 -3.4516100351236347e-17 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 8.2056361062066854e-17 -3.6214433702089572e-17 8.3333333333332565e-02 0.0000000000000000e+00 
1.8889437218922982e-16 -2.1179912487034412e-16 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 5.2855945355920649e-02 """

reference_mass = """1.0000000000000007e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 3.6429192995512949e-17 
0.0000000000000000e+00 1.0000000000000007e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -1.3010426069826053e-17 
0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000007e+00 -3.6429192995512949e-17 1.3010426069826053e-17 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -3.6429192995512949e-17 8.3333333333333245e-02 -2.6020852139652106e-18 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 1.3010426069826053e-17 -2.6020852139652106e-18 8.3333333333333259e-02 0.0000000000000000e+00 
3.6429192995512949e-17 -1.3010426069826053e-17 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.6666666666666671e-01"""
