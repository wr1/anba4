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

from dolfin import *


from anba4.voight_notation import (
    stressVectorToStressTensor,
    strainVectorToStrainTensor,
    strainTensorToStrainVector,
)
from anba4 import material


def pos3d(POS):
    "Return node coordinates Vector."
    return as_vector([POS[0], POS[1], 0.0])


def grad3d(u, up):
    "Return 3d gradient."
    g = grad(u)
    return as_tensor(
        [
            [g[0, 0], g[0, 1], up[0]],
            [g[1, 0], g[1, 1], up[1]],
            [g[2, 0], g[2, 1], up[2]],
        ]
    )


def epsilon(u, up):
    "Return symmetric 3D infinitesimal strain tensor."
    g3 = grad3d(u, up)
    return 0.5 * (g3.T + g3)


def rotated_epsilon(data, u, up):
    "Return symmetric 3D infinitesimal strain tensor rotated into material reference."
    eps = epsilon(u, up)
    rot = data["MaterialRotation_matrix"]
    rotMatrix = as_matrix(
        (
            (rot[0], rot[1], rot[2], rot[3], rot[4], rot[5]),
            (rot[6], rot[7], rot[8], rot[9], rot[10], rot[11]),
            (rot[12], rot[13], rot[14], rot[15], rot[16], rot[17]),
            (rot[18], rot[19], rot[20], rot[21], rot[22], rot[23]),
            (rot[24], rot[25], rot[26], rot[27], rot[28], rot[29]),
            (rot[30], rot[31], rot[32], rot[33], rot[34], rot[35]),
        )
    )
    roteps = strainVectorToStrainTensor(rotMatrix.T * strainTensorToStrainVector(eps))
    return roteps


def sigma_helper(mod, u, up):
    "Return second Piola-Kirchhoff stress tensor."
    et = epsilon(u, up)
    ev = strainTensorToStrainVector(et)
    elasticMatrix = as_matrix(
        (
            (mod[0], mod[1], mod[2], mod[3], mod[4], mod[5]),
            (mod[6], mod[7], mod[8], mod[9], mod[10], mod[11]),
            (mod[12], mod[13], mod[14], mod[15], mod[16], mod[17]),
            (mod[18], mod[19], mod[20], mod[21], mod[22], mod[23]),
            (mod[24], mod[25], mod[26], mod[27], mod[28], mod[29]),
            (mod[30], mod[31], mod[32], mod[33], mod[34], mod[35]),
        )
    )
    sv = elasticMatrix * ev
    st = stressVectorToStressTensor(sv)
    return st


def Sigma(data, u, up):
    "Return second Piola-Kirchhoff stress tensor."
    return sigma_helper(data["modulus"], u, up)


def RotatedSigma(data, u, up):
    "Return second Piola-Kirchhoff stress tensor."
    return sigma_helper(data["RotatedStress_modulus"], u, up)


def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver"""
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_) * dx
    b_proj = inner(v, v_) * dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return


def initialize_anba_model(
    mesh,
    degree,
    matLibrary,
    materials,
    plane_orientations,
    fiber_orientations,
    scaling_constraint=1.0,
):
    data = {}
    data["mesh"] = mesh
    data["degree"] = degree
    data["matLibrary"] = matLibrary
    data["materials"] = materials
    data["fiber_orientations"] = fiber_orientations
    data["plane_orientations"] = plane_orientations
    data["scaling_constraint"] = scaling_constraint
    data["modulus"] = CompiledExpression(
        material.ElasticModulus(
            matLibrary, materials, plane_orientations, fiber_orientations
        ),
        degree=0,
    )
    data["RotatedStress_modulus"] = CompiledExpression(
        material.RotatedStressElasticModulus(
            matLibrary, materials, plane_orientations, fiber_orientations
        ),
        degree=0,
    )
    data["MaterialRotation_matrix"] = CompiledExpression(
        material.TransformationMatrix(
            matLibrary, materials, plane_orientations, fiber_orientations
        ),
        degree=0,
    )
    data["density"] = CompiledExpression(
        material.MaterialDensity(matLibrary, materials), degree=0
    )
    data["POS"] = MeshCoordinates(mesh)
    return data
