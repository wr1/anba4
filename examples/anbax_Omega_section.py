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

import anba4
from anba4.io.export import export_model_json
import dolfin
import mshr

dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

E = 1.0
nu = 0.33
# Assmble into material mechanical property Matrix.
matMechanicProp = [E, nu]

Square1 = mshr.Rectangle(dolfin.Point(-40.0, 0.0, 0.0), dolfin.Point(40.0, 30.0, 0.0))
Square2 = mshr.Rectangle(dolfin.Point(-15.0, -10.0, 0.0), dolfin.Point(15.0, 25.0, 0.0))
Square3 = mshr.Rectangle(dolfin.Point(-50.0, 5.0, 0.0), dolfin.Point(-20.0, 30.0, 0.0))
Square4 = mshr.Rectangle(dolfin.Point(20.0, 5.0, 0.0), dolfin.Point(50.0, 30.0, 0.0))
C_shape = Square1 - Square2 - Square3 - Square4
mesh = mshr.generate_mesh(C_shape, 64)


dolfin.plot(mesh)

import matplotlib.pyplot as plt

plt.savefig("anbax_Omega_section_mesh.png")

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


input_data = anba4.InputData(
    mesh=mesh,
    degree=2,
    matLibrary=matLibrary,
    materials=materials,
    plane_orientations=plane_orientations,
    fiber_orientations=fiber_orientations,
)

anbax_data = anba4.initialize_anba_model(input_data)

export_model_json(input_data, "mesh_Omega_section.json")
anba4.initialize_fe_functions(anbax_data)
anba4.initialize_chains(anbax_data)
stiff = anba4.compute_stiffness(anbax_data)
stiff.view()

mass = anba4.compute_inertia(anbax_data)
mass.view()

output_file = "anbax_Omega_section.xdmf"
stress_result_file = dolfin.XDMFFile(output_file)
stress_result_file.parameters["functions_share_mesh"] = True
stress_result_file.parameters["rewrite_function_mesh"] = False
stress_result_file.parameters["flush_output"] = True

stress = anba4.stress_field(
    anbax_data,
    [
        1.0,
        0.0,
        0.0,
    ],
    [0.0, 0.0, 0.0],
    "local",
    "paraview",
)
stress_result_file.write(stress, t=0.0)

print(f"Stress field written to file: {output_file}")
