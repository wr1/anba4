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


def initialize_fe_functions(data):
    mesh = data.mesh
    degree = data.degree

    # Define function on space.
    UF3_ELEMENT = VectorElement("CG", mesh.ufl_cell(), degree, 3)
    data.UF3 = FunctionSpace(mesh, UF3_ELEMENT)

    # Lagrange multipliers needed to compute the stress resultants and moment resultants
    R3_ELEMENT = VectorElement("R", mesh.ufl_cell(), 0, 3)
    data.R3 = FunctionSpace(mesh, R3_ELEMENT)
    sp = parameters["reorder_dofs_serial"]
    parameters["reorder_dofs_serial"] = False
    R3R3_ELEMENT = MixedElement(R3_ELEMENT, R3_ELEMENT)
    data.R3R3 = FunctionSpace(mesh, R3R3_ELEMENT)
    parameters["reorder_dofs_serial"] = sp
    (data.RV3F, data.RV3M) = TestFunctions(data.R3R3)
    (data.RT3F, data.RT3M) = TrialFunctions(data.R3R3)

    STRESS_ELEMENT = VectorElement("DG", mesh.ufl_cell(), 0, 6)
    data.STRESS_FS = FunctionSpace(mesh, STRESS_ELEMENT)

    # Lagrange multipliers needed to impose the BCs
    R4_ELEMENT = VectorElement("R", mesh.ufl_cell(), 0, 4)
    data.R4 = FunctionSpace(mesh, R4_ELEMENT)
    sp = parameters["reorder_dofs_serial"]
    parameters["reorder_dofs_serial"] = False
    UF3R4_ELEMENT = MixedElement(UF3_ELEMENT, R4_ELEMENT)
    data.UF3R4 = FunctionSpace(mesh, UF3R4_ELEMENT)
    parameters["reorder_dofs_serial"] = sp

    data.UL = Function(data.UF3R4)
    (data.U, data.L) = split(data.UL)
    data.ULP = Function(data.UF3R4)
    (data.UP, data.LP) = split(data.ULP)
    data.ULV = TestFunction(data.UF3R4)
    (data.UV, data.LV) = TestFunctions(data.UF3R4)
    data.ULT = TrialFunction(data.UF3R4)
    (data.UT, data.LT) = TrialFunctions(data.UF3R4)
    return data
