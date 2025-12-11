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
#    You should have received a copy of the GNU General Public License
#    along with Anba.  If not, see <https://www.gnu.org/licenses/>.
#
# ----------------------------------------------------------------------
#

# import dolfin as df
import numpy as np

# from mshr import *
# from anba4 import *
import dolfin
import anba4
from anba4.io.export import export_model_json


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
E = [e_xx, e_yy, e_zz]
G = [g_yz, g_xz, g_xy]
nu = [nu_zy, nu_zx, nu_xy]
mat1 = anba4.material.OrthotropicMaterial(E, G, nu)
matLibrary = []
matLibrary.append(mat1)


input_data = anba4.InputData(
    mesh=mesh,
    degree=1,
    matLibrary=matLibrary,
    materials=materials,
    plane_orientations=plane_orientations,
    fiber_orientations=fiber_orientations,
    singular=True,
    scaling_constraint=1.0e9,
)

anbax_data = anba4.initialize_anba_model(input_data)

export_model_json(input_data, "mesh_rotation_singular.json")

anba4.initialize_fe_functions(anbax_data)
anba4.initialize_chains(anbax_data)
stiff = anba4.compute_stiffness(anbax_data)
stiff.view()
