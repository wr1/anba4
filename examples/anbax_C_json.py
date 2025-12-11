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

import dolfin
import anba4
from anba4.io.export import import_model_json


dolfin.parameters["form_compiler"]["optimize"] = True
dolfin.parameters["form_compiler"]["quadrature_degree"] = 2

# Load from the sample mesh.json or a generated one
serializable_input = import_model_json("mesh.json")

anbax_data = anba4.initialize_anba_model(serializable_input)
anba4.initialize_fe_functions(anbax_data)
anba4.initialize_chains(anbax_data)
stiff = anba4.compute_stiffness(anbax_data)
stiff.view()

mass = anba4.compute_inertia(anbax_data)
mass.view()

output_file = "anbax_C_section_json.xdmf"
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
strain = anba4.strain_field(
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
stress_result_file.write(strain, t=1.0)
print(f"Stress and strain fields written to {output_file}")
