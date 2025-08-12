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


def initialize_fe_basis(data):
    mesh = data["mesh"]
    degree = data["degree"]

    # Define function on space.
    UF3_ELEMENT = VectorElement("CG", mesh.ufl_cell(), degree, 3)
    data["UF3"] = FunctionSpace(mesh, UF3_ELEMENT)

    # Lagrange multipliers needed to compute the stress resultants and moment resultants
    R3_ELEMENT = VectorElement("R", mesh.ufl_cell(), 0, 3)
    data["R3"] = FunctionSpace(mesh, R3_ELEMENT)
    sp = parameters["reorder_dofs_serial"]
    parameters["reorder_dofs_serial"] = False
    R3R3_ELEMENT = MixedElement(R3_ELEMENT, R3_ELEMENT)
    data["R3R3"] = FunctionSpace(mesh, R3R3_ELEMENT)
    parameters["reorder_dofs_serial"] = sp
    (data["RV3F"], data["RV3M"]) = TestFunctions(data["R3R3"])
    (data["RT3F"], data["RT3M"]) = TrialFunctions(data["R3R3"])

    STRESS_ELEMENT = VectorElement("DG", mesh.ufl_cell(), 0, 6)
    data["STRESS_FS"] = FunctionSpace(mesh, STRESS_ELEMENT)

    # Lagrange multipliers needed to impose the BCs
    R4_ELEMENT = VectorElement("R", mesh.ufl_cell(), 0, 4)
    data["R4"] = FunctionSpace(mesh, R4_ELEMENT)
    sp = parameters["reorder_dofs_serial"]
    parameters["reorder_dofs_serial"] = False
    UF3R4_ELEMENT = MixedElement(UF3_ELEMENT, R4_ELEMENT)
    data["UF3R4"] = FunctionSpace(mesh, UF3R4_ELEMENT)
    parameters["reorder_dofs_serial"] = sp

    data["UL"] = Function(data["UF3R4"])
    (data["U"], data["L"]) = split(data["UL"])
    data["ULP"] = Function(data["UF3R4"])
    (data["UP"], data["LP"]) = split(data["ULP"])
    data["ULV"] = TestFunction(data["UF3R4"])
    (data["UV"], data["LV"]) = TestFunctions(data["UF3R4"])
    data["ULT"] = TrialFunction(data["UF3R4"])
    (data["UT"], data["LT"]) = TrialFunctions(data["UF3R4"])

    data["base_chains_expression"] = []
    data["linear_chains_expression"] = []
    data["Torsion"] = Expression(
        ("-x[1]", "x[0]", "0.", "0.", "0.", "0.", "0."),
        element=data["UF3R4"].ufl_element(),
    )
    data["Flex_y"] = Expression(
        ("0.", "0.", "-x[0]", "0.", "0.", "0.", "0."),
        element=data["UF3R4"].ufl_element(),
    )
    data["Flex_x"] = Expression(
        ("0.", "0.", "-x[1]", "0.", "0.", "0.", "0."),
        element=data["UF3R4"].ufl_element(),
    )

    data["base_chains_expression"].append(Constant((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)))
    data["base_chains_expression"].append(data["Torsion"])
    data["base_chains_expression"].append(Constant((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    data["base_chains_expression"].append(Constant((0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    data["linear_chains_expression"].append(data["Flex_y"])
    data["linear_chains_expression"].append(data["Flex_x"])

    data["chains"] = [[], [], [], []]
    data["chains_d"] = [[], [], [], []]
    data["chains_l"] = [[], [], [], []]

    # fill chains
    for i in range(4):
        for k in range(2):
            data["chains"][i].append(Function(data["UF3R4"]))
    for i in range(2, 4):
        for k in range(2):
            data["chains"][i].append(Function(data["UF3R4"]))

    # initialize constant chains
    for i in range(4):
        data["chains"][i][0].interpolate(data["base_chains_expression"][i])
    # keep torsion independent from translation
    for i in [0, 2, 3]:
        k = (data["chains"][1][0].vector().inner(data["chains"][i][0].vector())) / (
            data["chains"][i][0].vector().inner(data["chains"][i][0].vector())
        )
        data["chains"][1][0].vector()[:] -= k * data["chains"][i][0].vector()

    # unit norm chains
    tmpnorm = []
    for i in range(4):
        tmpnorm.append(data["chains"][i][0].vector().norm("l2"))
        data["chains"][i][0].vector()[:] *= 1.0 / tmpnorm[i]
    # null space
    data["null_space"] = VectorSpaceBasis(
        [data["chains"][i][0].vector() for i in range(4)]
    )

    # initialize linear chains
    for i in range(2, 4):
        data["chains"][i][1].interpolate(data["linear_chains_expression"][i - 2])
        data["chains"][i][1].vector()[:] *= 1.0 / tmpnorm[i]
        data["null_space"].orthogonalize(data["chains"][i][1].vector())
    del tmpnorm

    for i in range(4):
        for k in range(2):
            (d, l) = split(data["chains"][i][k])
            data["chains_d"][i].append(d)
            data["chains_l"][i].append(l)

    for i in range(2, 4):
        for k in range(2, 4):
            (d, l) = split(data["chains"][i][k])
            data["chains_d"][i].append(d)
            data["chains_l"][i].append(l)
    return data
