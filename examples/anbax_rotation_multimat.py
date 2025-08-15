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

from anba4.io.export import export_model_json
import dolfin
import numpy as np
import anba4

dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

# Basic material parameters. 9 is needed for orthotropic materials.

e_xx = 9.8e9
e_yy = 9.8e9
e_zz = 1.42e11
g_xy = 4.8e9
g_xz = 6.0e9
g_yz = 6.0e9
nu_xy = 0.34
nu_zx = 0.3
nu_zy = 0.3
# Assmble into material mechanical property Matrix.
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

matMechanicProp1 = [9.8e9 / 100.0, 0.3]
# Meshing domain.
sectionWidth = 3.0023e-2
sectionHeight = 1.9215e-3
nPly = 16  # t = 0.2452mm per ply.
# mesh = RectangleMesh.create([Point(0., 0.), Point(sectionWidth, sectionHeight)], [30, 32], CellType.Type.quadrilateral)
mesh = dolfin.RectangleMesh(
    dolfin.Point(0.0, 0.0), dolfin.Point(sectionWidth, sectionHeight), 30, 32, "crossed"
)
dolfin.ALE.move(mesh, dolfin.Constant([-sectionWidth / 2.0, -sectionHeight / 2.0]))
mesh_points = mesh.coordinates()
# print(mesh_points)

# CompiledSubDomain
materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
# isActive = MeshFunction("bool", mesh, mesh.topology().dim())
tol = 1e-14

"""subdomain_0 = CompiledSubDomain(["x[1] >= -5.0*thickness + tol && x[1] <= -4.0*thickness + tol",\
                                 "x[1] >= -3.0*thickness + tol && x[1] <= -2.0*thickness + tol",\
                                 "x[1] >= -1.0*thickness + tol && x[1] <= 0.0*thickness + tol"\
                                 "x[1] >= 0.0*thickness + tol && x[1] <= 1.0*thickness + tol",\
                                 "x[1] >= 2.0*thickness + tol && x[1] <= 3.0*thickness + tol",\
                                 "x[1] >= 4.0*thickness + tol && x[1] <= 5.0*thickness + tol"], \
                                 thickness = sectionHeight / nPly, tol=tol)
subdomain_1 = CompiledSubDomain(["x[1] >= -6.0*thickness + tol && x[1] <= -5.0*thickness + tol",\
                                 "x[1] >= -4.0*thickness + tol && x[1] <= -3.0*thickness + tol",\
                                 "x[1] >= -2.0*thickness + tol && x[1] <= -1.0*thickness + tol",\
                                 "x[1] >= 1.0*thickness + tol && x[1] <= 2.0*thickness + tol",\
                                 "x[1] >= 3.0*thickness + tol && x[1] <= 4.0*thickness + tol",\
                                 "x[1] >= 5.0*thickness + tol && x[1] <= 6.0*thickness + tol"],\
                                  thickness = sectionHeight / nPly, tol=tol)
                                  """
subdomain_0_p20 = dolfin.CompiledSubDomain(
    "x[1] >= -8.0*thickness - tol && x[1] <= -7.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_1_m70 = dolfin.CompiledSubDomain(
    "x[1] >= -7.0*thickness - tol && x[1] <= -6.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_2_m70 = dolfin.CompiledSubDomain(
    "x[1] >= -6.0*thickness - tol && x[1] <= -5.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_3_p20 = dolfin.CompiledSubDomain(
    "x[1] >= -5.0*thickness - tol && x[1] <= -4.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)

subdomain_4_p20 = dolfin.CompiledSubDomain(
    "x[1] >= -4.0*thickness - tol && x[1] <= -3.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_5_m70 = dolfin.CompiledSubDomain(
    "x[1] >= -3.0*thickness - tol && x[1] <= -2.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_6_m70 = dolfin.CompiledSubDomain(
    "x[1] >= -2.0*thickness - tol && x[1] <= -1.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_7_p20 = dolfin.CompiledSubDomain(
    "x[1] >= -1.0*thickness - tol && x[1] <= -0.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)

subdomain_8_m20 = dolfin.CompiledSubDomain(
    "x[1] >= 0.0*thickness - tol && x[1] <= 1.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_9_p70 = dolfin.CompiledSubDomain(
    "x[1] >= 1.0*thickness - tol && x[1] <= 2.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_10_p70 = dolfin.CompiledSubDomain(
    "x[1] >= 2.0*thickness - tol && x[1] <= 3.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_11_m20 = dolfin.CompiledSubDomain(
    "x[1] >= 3.0*thickness - tol && x[1] <= 4.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)

subdomain_12_m20 = dolfin.CompiledSubDomain(
    "x[1] >= 4.0*thickness - tol && x[1] <= 5.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_13_p70 = dolfin.CompiledSubDomain(
    "x[1] >= 5.0*thickness - tol && x[1] <= 6.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_14_p70 = dolfin.CompiledSubDomain(
    "x[1] >= 6.0*thickness - tol && x[1] <= 7.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)
subdomain_15_m20 = dolfin.CompiledSubDomain(
    "x[1] >= 7.0*thickness - tol && x[1] <= 8.0*thickness + tol",
    thickness=sectionHeight / nPly,
    tol=tol,
)

# Rotate mesh.
rotation_angle = 23.0
materials.set_all(0)
fiber_orientations.set_all(0.0)
plane_orientations.set_all(rotation_angle)

subdomain_0_p20.mark(materials, 1)

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

# rotate mesh.
rotate = dolfin.Expression(
    (
        "x[0] * (cos(rotation_angle)-1.0) - x[1] * sin(rotation_angle)",
        "x[0] * sin(rotation_angle) + x[1] * (cos(rotation_angle)-1.0)",
    ),
    rotation_angle=rotation_angle * np.pi / 180.0,
    degree=1,
)

dolfin.ALE.move(mesh, rotate)

# Build material property library.
mat1 = anba4.material.OrthotropicMaterial(matMechanicProp)
mat2 = anba4.material.IsotropicMaterial(matMechanicProp1)
matLibrary = []
matLibrary.append(mat1)
matLibrary.append(mat2)


input_data = anba4.InputData(
    mesh=mesh,
    degree=1,
    matLibrary=matLibrary,
    materials=materials,
    plane_orientations=plane_orientations,
    fiber_orientations=fiber_orientations,
    scaling_constraint=1.0e9,
)

anbax_data = anba4.initialize_anba_model(input_data)

export_model_json(input_data, "mesh_rotation_multimat.json")

anba4.initialize_fe_functions(anbax_data)
anba4.initialize_chains(anbax_data)
stiff = anba4.compute_stiffness(anbax_data)
stiff.view()

if __name__ == "__test__":
    np.testing.assert_almost_equal(
        stiff.getValues(range(6), range(6)),
        np.array(
            [
                [
                    4.77322779e05,
                    1.87291500e05,
                    -1.00040147e05,
                    -2.65592457e02,
                    -1.11914084e02,
                    -5.56374255e01,
                ],
                [
                    1.87291500e05,
                    1.15592180e05,
                    -4.24645230e04,
                    -1.11914078e02,
                    -4.94441224e01,
                    -2.36166857e01,
                ],
                [
                    -1.00040147e05,
                    -4.24645230e04,
                    3.00795578e06,
                    3.10776505e02,
                    1.31916800e02,
                    7.41818375e02,
                ],
                [
                    -2.65592457e02,
                    -1.11914078e02,
                    3.10776505e02,
                    3.38577775e01,
                    -7.79421592e01,
                    1.59495464e-01,
                ],
                [
                    -1.11914084e02,
                    -4.94441224e01,
                    1.31916800e02,
                    -7.79421592e01,
                    1.84393514e02,
                    6.77018038e-02,
                ],
                [
                    -5.56374255e01,
                    -2.36166857e01,
                    7.41818375e02,
                    1.59495464e-01,
                    6.77018038e-02,
                    8.30649734e-01,
                ],
            ]
        ),
        3,
    )
