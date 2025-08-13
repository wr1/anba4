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
    Square2 = mshr.Rectangle(Point(thickness, -1 + thickness, 0), Point(2.0, 1.0 - thickness, 0))
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
    np.testing.assert_allclose(decoupled_stiff[0:3, 3:6], np.zeros((3,3)), atol=1e-6)
    np.testing.assert_allclose(decoupled_stiff[3:6, 0:3], np.zeros((3,3)), atol=1e-6)

