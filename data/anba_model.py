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
from anba4 import material
from anba4.data_model import AnbaData


def initialize_anba_model(
    mesh,
    degree,
    matLibrary,
    materials,
    plane_orientations,
    fiber_orientations,
    scaling_constraint=1.0,
):
    data = AnbaData(
        mesh=mesh,
        degree=degree,
        matLibrary=matLibrary,
        materials=materials,
        fiber_orientations=fiber_orientations,
        plane_orientations=plane_orientations,
        scaling_constraint=scaling_constraint,
    )
    data.modulus = CompiledExpression(
        material.ElasticModulus(
            matLibrary, materials, plane_orientations, fiber_orientations
        ),
        degree=0,
    )
    data.RotatedStress_modulus = CompiledExpression(
        material.RotatedStressElasticModulus(
            matLibrary, materials, plane_orientations, fiber_orientations
        ),
        degree=0,
    )
    data.MaterialRotation_matrix = CompiledExpression(
        material.TransformationMatrix(
            matLibrary, materials, plane_orientations, fiber_orientations
        ),
        degree=0,
    )
    data.density = CompiledExpression(
        material.MaterialDensity(matLibrary, materials), degree=0
    )
    data.POS = MeshCoordinates(mesh)
    return data
