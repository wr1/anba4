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


def initialize_chains(data):
    data.base_chains_expression = []
    data.linear_chains_expression = []
    data.Torsion = Expression(
        ("-x[1]", "x[0]", "0.", "0.", "0.", "0.", "0."),
        element=data.UF3R4.ufl_element(),
    )
    data.Flex_y = Expression(
        ("0.", "0.", "-x[0]", "0.", "0.", "0.", "0."), element=data.UF3R4.ufl_element()
    )
    data.Flex_x = Expression(
        ("0.", "0.", "-x[1]", "0.", "0.", "0.", "0."), element=data.UF3R4.ufl_element()
    )

    data.base_chains_expression.append(Constant((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)))
    data.base_chains_expression.append(data.Torsion)
    data.base_chains_expression.append(Constant((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    data.base_chains_expression.append(Constant((0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    data.linear_chains_expression.append(data.Flex_y)
    data.linear_chains_expression.append(data.Flex_x)

    data.chains = [[], [], [], []]
    data.chains_d = [[], [], [], []]
    data.chains_l = [[], [], [], []]

    # fill chains
    for i in range(4):
        for k in range(2):
            data.chains[i].append(Function(data.UF3R4))
    for i in range(2, 4):
        for k in range(2):
            data.chains[i].append(Function(data.UF3R4))

    # initialize constant chains
    for i in range(4):
        data.chains[i][0].interpolate(data.base_chains_expression[i])
    # keep torsion independent from translation
    for i in [0, 2, 3]:
        k = (data.chains[1][0].vector().inner(data.chains[i][0].vector())) / (
            data.chains[i][0].vector().inner(data.chains[i][0].vector())
        )
        data.chains[1][0].vector()[:] -= k * data.chains[i][0].vector()

    # unit norm chains
    tmpnorm = []
    for i in range(4):
        tmpnorm.append(data.chains[i][0].vector().norm("l2"))
        data.chains[i][0].vector()[:] *= 1.0 / tmpnorm[i]
    # null space
    data.null_space = VectorSpaceBasis([data.chains[i][0].vector() for i in range(4)])

    # initialize linear chains
    for i in range(2, 4):
        data.chains[i][1].interpolate(data.linear_chains_expression[i - 2])
        data.chains[i][1].vector()[:] *= 1.0 / tmpnorm[i]
        data.null_space.orthogonalize(data.chains[i][1].vector())
    del tmpnorm

    for i in range(4):
        for k in range(2):
            (d, l) = split(data.chains[i][k])
            data.chains_d[i].append(d)
            data.chains_l[i].append(l)

    for i in range(2, 4):
        for k in range(2, 4):
            (d, l) = split(data.chains[i][k])
            data.chains_d[i].append(d)
            data.chains_l[i].append(l)
    return data
