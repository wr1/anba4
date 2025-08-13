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
from .. import material
from ..data_model import (
    AnbaData,
    InputData,
    FEFunctions,
    Chains,
    OutputData,
    MaterialData,
)


def initialize_anba_model(
    mesh,
    degree,
    matLibrary,
    materials,
    plane_orientations,
    fiber_orientations,
    scaling_constraint=1.0,
    singular=False,
):
    input_data = InputData(
        mesh=mesh,
        degree=degree,
        matLibrary=matLibrary,
        materials=materials,
        fiber_orientations=fiber_orientations,
        plane_orientations=plane_orientations,
        scaling_constraint=scaling_constraint,
        singular=singular,
    )
    fe_functions = FEFunctions(POS=MeshCoordinates(mesh))
    data = AnbaData(
        input_data=input_data,
        fe_functions=fe_functions,
        chains=Chains(),
        output_data=OutputData(),
        material_data=MaterialData(),
    )
    data.material_data.modulus = material.ElasticModulus(
        matLibrary, materials, plane_orientations, fiber_orientations, degree=0
    )
    data.material_data.RotatedStress_modulus = material.RotatedStressElasticModulus(
        matLibrary, materials, plane_orientations, fiber_orientations, degree=0
    )
    data.material_data.MaterialRotation_matrix = material.TransformationMatrix(
        matLibrary, materials, plane_orientations, fiber_orientations, degree=0
    )
    data.material_data.density = material.MaterialDensity(
        matLibrary, materials, degree=0
    )
    return data
