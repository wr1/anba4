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
from petsc4py import PETSc

from anba4.utils import pos3d


def compute_inertia(data):
    Mf = dot(data.RV3F, data.RT3F) * data.density[0] * dx
    Mf -= dot(data.RV3F, cross(pos3d(data.POS), data.RT3M)) * data.density[0] * dx
    Mf -= dot(cross(pos3d(data.POS), data.RV3M), data.RT3F) * data.density[0] * dx
    Mf += (
        dot(cross(pos3d(data.POS), data.RV3M), cross(pos3d(data.POS), data.RT3M))
        * data.density[0]
        * dx
    )
    MM = assemble(Mf)
    M = as_backend_type(MM).mat()
    Mass = PETSc.Mat()
    M.convert("dense", Mass)
    return Mass
