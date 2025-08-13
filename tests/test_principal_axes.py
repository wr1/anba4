import pytest
import numpy as np
from dolfin import *
from anba4 import *
import mshr


def test_principal_axes():
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["quadrature_degree"] = 2
    E = 1.0
    nu = 0.33
    matMechanicProp = [E, nu]
    thickness = 0.1
    Square1 = mshr.Rectangle(Point(0.0, -1.0, 0.0), Point(1.0, 1.0, 0.0))
    Square2 = mshr.Rectangle(
        Point(thickness, -1 + thickness, 0), Point(2.0, 1.0 - thickness, 0)
    )
    C_shape = Square1 - Square2
    mesh = mshr.generate_mesh(C_shape, 64)
    rot_angle = 30.0 / 180.0 * np.pi
    cr = np.cos(rot_angle)
    sr = np.sin(rot_angle)
    rot_tensor = np.array([[cr, -sr], [sr, cr]])
    mesh.coordinates()[:] = (rot_tensor @ mesh.coordinates().T).T
    mesh.coordinates()[:] += np.array([3, 1])
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(90.0)
    mat1 = material.IsotropicMaterial(matMechanicProp, 1.0)
    matLibrary = [mat1]
    anbax_data = initialize_anba_model(
        mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations
    )
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    stiff_mat = stiff.getValues(range(6), range(6))
    decoupled_stiff = utils.DecoupleStiffness(stiff_mat)
    angle = utils.PrincipalAxesRotationAngle(decoupled_stiff)
    np.testing.assert_allclose(angle * 180 / np.pi, 30, atol=1e-3)
    np.testing.assert_allclose(decoupled_stiff[0:3, 3:6], np.zeros((3, 3)), atol=1e-6)
    np.testing.assert_allclose(decoupled_stiff[3:6, 0:3], np.zeros((3, 3)), atol=1e-6)


reference_stiffness = """4.9983692051719798e-02 -6.4216083302349259e-03 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -6.0014130758270327e-02 
-6.4216083302310063e-03 5.7386051916386377e-02 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.6258740045203857e-01 
0.0000000000000000e+00 0.0000000000000000e+00 3.7999999999938078e-01 4.3449999999926625e-01 -1.2343967690104505e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 4.3449999999926836e-01 6.7776666666571550e-01 -1.3277487113035373e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -1.2343967690104500e+00 -1.3277487113035293e+00 4.0941472807347932e+00 0.0000000000000000e+00 
-6.0014130758255915e-02 1.6258740045203884e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 4.9662070024812510e-01 """
reference_mass = """3.7999999999999967e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -4.3450000000000011e-01 
0.0000000000000000e+00 3.7999999999999967e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.2343967690125053e+00 
0.0000000000000000e+00 0.0000000000000000e+00 3.7999999999999967e-01 4.3450000000000011e-01 -1.2343967690125053e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 4.3450000000000011e-01 6.7776666666666541e-01 -1.3277487113059632e+00 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -1.2343967690125053e+00 -1.3277487113059632e+00 4.0941472807416801e+00 0.0000000000000000e+00 
-4.3450000000000011e-01 1.2343967690125053e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 4.7719139474083487e+00 """

decoupled_stiffness = """ 4.99836921e-02 -6.42160833e-03  0.00000000e+00  0.00000000e+00   0.00000000e+00  6.93889390e-18
   -6.42160833e-03  5.73860519e-02  0.00000000e+00  0.00000000e+00   0.00000000e+00 -2.77555756e-17
    0.00000000e+00  0.00000000e+00  3.80000000e-01 -5.55111512e-17   0.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00  2.05391260e-15  1.80950219e-01   8.36865417e-02  0.00000000e+00
    0.00000000e+00  0.00000000e+00  4.44089210e-16  8.36865417e-02   8.43173246e-02  0.00000000e+00
    3.68455266e-15 -3.08086889e-15  0.00000000e+00  0.00000000e+00   0.00000000e+00  4.74048372e-04"""
rotation_angle = 29.99999999999973
