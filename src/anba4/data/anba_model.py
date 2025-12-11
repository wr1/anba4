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
import numpy as np
from typing import Union, List
from .. import material
from .data_model import (
    AnbaData,
    InputData,
    FEFunctions,
    Chains,
    OutputData,
    MaterialData,
    SerializableInputData,
)


def initialize_anba_model(
    input_data: Union[InputData, SerializableInputData],
) -> AnbaData:
    if isinstance(input_data, SerializableInputData):
        # Construct Dolfin mesh from points and cells using MeshEditor
        coordinates = np.array(
            [p[:2] for p in input_data.points]
        )  # Assume 2D, ignore z
        cells = np.array(input_data.cells, dtype="uintp")

        mesh = Mesh()
        editor = MeshEditor()
        editor.open(mesh, "triangle", 2, 2)  # tdim=2, gdim=2
        editor.init_vertices(coordinates.shape[0])
        for i, coord in enumerate(coordinates):
            editor.add_vertex(i, coord)
        editor.init_cells(cells.shape[0])
        for i, cell in enumerate(cells):
            editor.add_cell(i, cell)
        editor.close()

        # Reconstruct matLibrary from dicts
        matLibrary: List[material.Material] = []
        for md in input_data.mat_library:
            if md["type"] == "isotropic":
                matLibrary.append(material.IsotropicMaterial.from_dict(md))
            elif md["type"] == "orthotropic":
                matLibrary.append(material.OrthotropicMaterial.from_dict(md))
            else:
                raise ValueError(f"Unknown material type: {md['type']}")

        # Create MeshFunctions
        materials = MeshFunction("size_t", mesh, mesh.topology().dim())
        materials.array()[:] = np.array(input_data.material_ids, dtype=np.uintp)

        fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
        fiber_orientations.array()[:] = np.array(input_data.fiber_orientations)

        plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())
        plane_orientations.array()[:] = np.array(input_data.plane_orientations)

        # Create InputData with Dolfin objects
        dolfin_input = InputData(
            mesh=mesh,
            degree=input_data.degree,
            matLibrary=matLibrary,
            materials=materials,
            fiber_orientations=fiber_orientations,
            plane_orientations=plane_orientations,
            scaling_constraint=input_data.scaling_constraint,
            singular=input_data.singular,
        )
    else:
        dolfin_input = input_data

    fe_functions = FEFunctions(POS=MeshCoordinates(dolfin_input.mesh))
    data = AnbaData(
        input_data=dolfin_input,
        fe_functions=fe_functions,
        chains=Chains(),
        output_data=OutputData(),
        material_data=MaterialData(),
    )
    data.material_data.modulus = material.ElasticModulus(
        dolfin_input.matLibrary,
        dolfin_input.materials,
        dolfin_input.plane_orientations,
        dolfin_input.fiber_orientations,
        degree=0,
    )
    data.material_data.RotatedStress_modulus = material.RotatedStressElasticModulus(
        dolfin_input.matLibrary,
        dolfin_input.materials,
        dolfin_input.plane_orientations,
        dolfin_input.fiber_orientations,
        degree=0,
    )
    data.material_data.MaterialRotation_matrix = material.TransformationMatrix(
        dolfin_input.matLibrary,
        dolfin_input.materials,
        dolfin_input.plane_orientations,
        dolfin_input.fiber_orientations,
        degree=0,
    )
    data.material_data.density = material.MaterialDensity(
        dolfin_input.matLibrary, dolfin_input.materials, degree=0
    )
    return data
