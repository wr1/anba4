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

df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.
matMechanicProp1 = [80000, 0.3]
matMechanicProp2 = [80000 * 0.5, 0.3]
matMechanicProp3 = [80000 * 0.00001, 0.3]

# Meshing domain.
sectionWidth = 20
sectionHeight = 20

# mesh = RectangleMesh.create([Point(0., 0.), Point(sectionWidth, sectionHeight)], [30, 32], CellType.Type.quadrilateral)
mesh = df.RectangleMesh(
    df.Point(0.0, 0.0), df.Point(sectionWidth, sectionHeight), 50, 50, "crossed"
)
df.ALE.move(mesh, df.Constant([-sectionWidth / 2.0, -sectionHeight / 2.0]))

# CompiledSubDomain
materials = df.MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = df.MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = df.MeshFunction("double", mesh, mesh.topology().dim())
# isActive = MeshFunction("bool", mesh, mesh.topology().dim())
tol = 1e-14

lower_portion = df.CompiledSubDomain("x[1] <= 0 + tol", tol=tol)
hole = df.CompiledSubDomain(
    "(x[1] >= -6 + tol && x[1] <= 6. + tol)&&(x[0] >= -2 + tol && x[0] <= 2. + tol)",
    tol=tol,
)

# Rotate mesh.
rotation_angle = 0.0
materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(rotation_angle)

lower_portion.mark(materials, 1)
hole.mark(materials, 2)
plot(materials, "Subdomains")
# plt.show()

# rotate mesh.
mat1 = IsotropicMaterial(matMechanicProp1)
mat2 = IsotropicMaterial(matMechanicProp2)
mat3 = IsotropicMaterial(matMechanicProp3)
matLibrary = []
matLibrary.append(mat1)
matLibrary.append(mat2)
matLibrary.append(mat3)


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
stiff.view()

JordanChains = df.XDMFFile("jordan_chains_with_real_hole.xdmf")
JordanChains.parameters["functions_share_mesh"] = True
JordanChains.parameters["rewrite_function_mesh"] = False
JordanChains.parameters["flush_output"] = True
for i in range(len(anbax_data.chains.chains_d)):
    for j in range(len(anbax_data.chains.chains_d[i])):
        # print('chain_'+str(i)+'_'+str(j))
        chain = df.Function(
            anbax_data.fe_functions.UF3, name="chain_" + str(i) + "_" + str(j)
        )
        chain.vector()[:] = df.project(
            anbax_data.chains.chains_d[i][j], anbax_data.fe_functions.UF3
        ).vector()
        JordanChains.write(chain, t=0.0)
