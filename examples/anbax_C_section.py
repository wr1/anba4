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

import dolfin as df
from anba4 import *
import mshr

df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

E = 1.0
nu = 0.33
# Assmble into material mechanical property Matrix.
matMechanicProp = [E, nu]
# Meshing domain.

thickness = 0.1
Square1 = mshr.Rectangle(df.Point(0.0, -1.0, 0.0), df.Point(1.0, 1.0, 0.0))
Square2 = mshr.Rectangle(
    df.Point(thickness, -1 + thickness, 0), df.Point(2.0, 1.0 - thickness, 0)
)
C_shape = Square1 - Square2
mesh = mshr.generate_mesh(C_shape, 64)

# CompiledSubDomain
materials = df.MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = df.MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = df.MeshFunction("double", mesh, mesh.topology().dim())

materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(90.0)

# Build material property library.
mat1 = IsotropicMaterial(matMechanicProp, 1.0)

matLibrary = []
matLibrary.append(mat1)

anbax_data = initialize_anba_model(
    mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations
)
initialize_fe_functions(anbax_data)
initialize_chains(anbax_data)
stiff = compute_stiffness(anbax_data)
stiff.view()

mass = compute_inertia(anbax_data)
mass.view()

stress_result_file = df.XDMFFile("Stress.xdmf")
stress_result_file.parameters["functions_share_mesh"] = True
stress_result_file.parameters["rewrite_function_mesh"] = False
stress_result_file.parameters["flush_output"] = True

# anba.stress_field([1., 0., 0.,], [0., 0., 0.], "local", "paraview")
# anba.strain_field([1., 0., 0.,], [0., 0., 0.], "local", "paraview")
# stress_result_file.write(anba.STRESS, t = 0.)
# stress_result_file.write(anba.STRAIN, t = 1.)
