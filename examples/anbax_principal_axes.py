#
# Copyright (C) 2018 Marco Morandini
#
# ----------------------------------------------------------------------
#
#    This file is part of Anba.
#
#    Anba is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Anba is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Anba.  If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------
#

import dolfin
import numpy as np
import anba4
import mshr

dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

E = 1.0
nu = 0.33
# Assmble into material mechanical property Matrix.
matMechanicProp = [E, nu]
# Meshing domain.

thickness = 0.1
Square1 = mshr.Rectangle(dolfin.Point(0.0, -1.0, 0.0), dolfin.Point(1.0, 1.0, 0.0))
Square2 = mshr.Rectangle(
    dolfin.Point(thickness, -1 + thickness, 0), dolfin.Point(2.0, 1.0 - thickness, 0)
)
C_shape = Square1 - Square2
mesh = mshr.generate_mesh(C_shape, 64)
rot_angle = 30.0 / 180.0 * np.pi
cr = np.cos(rot_angle)
sr = np.sin(rot_angle)
rot_tensor = np.array([[cr, -sr], [sr, cr]])
mesh.coordinates()[:] = (rot_tensor @ mesh.coordinates().T).T
mesh.coordinates()[:] += np.array([3, 1])

dolfin.plot(mesh)

import matplotlib.pyplot as plt

plt.savefig("anbax_principal_axes_mesh.png")


# CompiledSubDomain
materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())

materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(90.0)

# Build material property library.
mat1 = anba4.material.IsotropicMaterial(matMechanicProp, 1.0)

matLibrary = []
matLibrary.append(mat1)

anbax_data = anba4.initialize_anba_model(
    mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations
)
anba4.initialize_fe_functions(anbax_data)
anba4.initialize_chains(anbax_data)

stiff = anba4.compute_stiffness(anbax_data)
print("Stiff:")
stiff.view()

mass = anba4.compute_inertia(anbax_data)
print("Mass:")
mass.view()


stiff_mat = stiff.getValues(range(6), range(6))
decoupled_stiff = anba4.utils.DecoupleStiffness(stiff_mat)
print("Decoupled Stiff:")
print(decoupled_stiff)
angle = anba4.utils.PrincipalAxesRotationAngle(decoupled_stiff)
print("Rotation angle:", angle / np.pi * 180.0)
