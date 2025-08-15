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

# import dolfin as df
# from anba4 import *
from anba4.io.export import export_model_json
import dolfin
import anba4
import mshr

dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.
matMechanicProp1 = [80000, 0.3]
matMechanicProp2 = [80000 * 0.5, 0.3]
matMechanicProp3 = [80000 * 0.001, 0.3]

# Meshing domain.
sectionWidth = 20
sectionHeight = 20


# mesh = RectangleMesh.create([Point(0., 0.), Point(sectionWidth, sectionHeight)], [30, 32], CellType.Type.quadrilateral)
Square = mshr.Rectangle(dolfin.Point(-10.0, -10.0), dolfin.Point(10.0, 10.0))
Rectangle = mshr.Rectangle(dolfin.Point(-2.0, -6.0), dolfin.Point(2.0, 6.0))
Domain = Square - Rectangle
mesh = mshr.generate_mesh(Domain, 64)

# CompiledSubDomain
materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
# isActive = MeshFunction("bool", mesh, mesh.topology().dim())
tol = 1e-14

lower_portion = dolfin.CompiledSubDomain("x[1] <= 0 + tol", tol=tol)

# Rotate mesh.
rotation_angle = 0.0
materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(rotation_angle)

lower_portion.mark(materials, 1)
dolfin.plot(materials, "Subdomains")
# plt.show()

# rotate mesh.
mat1 = anba4.material.IsotropicMaterial(matMechanicProp1)
mat2 = anba4.material.IsotropicMaterial(matMechanicProp2)
mat3 = anba4.material.IsotropicMaterial(matMechanicProp3)
matLibrary = []
matLibrary.append(mat1)
matLibrary.append(mat2)
matLibrary.append(mat3)

input_data = anba4.InputData(
    mesh=mesh,
    degree=1,
    matLibrary=matLibrary,
    materials=materials,
    plane_orientations=plane_orientations,
    fiber_orientations=fiber_orientations,
    scaling_constraint=1,
)
export_model_json(input_data, "mesh_multimat_with_hole2.json")

anbax_data = anba4.initialize_anba_model(input_data)


anba4.initialize_fe_functions(anbax_data)
anba4.initialize_chains(anbax_data)
stiff = anba4.compute_stiffness(anbax_data)
stiff.view()

output_file = "anbax_multimat_with_hole2.xdmf"
JordanChains = dolfin.XDMFFile(output_file)
JordanChains.parameters["functions_share_mesh"] = True
JordanChains.parameters["rewrite_function_mesh"] = False
JordanChains.parameters["flush_output"] = True
for i in range(len(anbax_data.chains.chains_d)):
    for j in range(len(anbax_data.chains.chains_d[i])):
        # print('chain_'+str(i)+'_'+str(j))
        chain = dolfin.Function(
            anbax_data.fe_functions.UF3, name="chain_" + str(i) + "_" + str(j)
        )
        chain.vector()[:] = dolfin.project(
            anbax_data.chains.chains_d[i][j], anbax_data.fe_functions.UF3
        ).vector()
        JordanChains.write(chain, t=0.0)


print(f"Output written to file: {output_file}")
