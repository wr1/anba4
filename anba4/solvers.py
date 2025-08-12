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

import numpy as np

from anba4.anba_model import (
    Sigma,
    RotatedSigma,
    epsilon,
    rotated_epsilon,
    local_project,
    pos3d,
)


def compute_inertia(data):
    Mf = dot(data["RV3F"], data["RT3F"]) * data["density"][0] * dx
    Mf -= (
        dot(data["RV3F"], cross(pos3d(data["POS"]), data["RT3M"]))
        * data["density"][0]
        * dx
    )
    Mf -= (
        dot(cross(pos3d(data["POS"]), data["RV3M"]), data["RT3F"])
        * data["density"][0]
        * dx
    )
    Mf += (
        dot(
            cross(pos3d(data["POS"]), data["RV3M"]),
            cross(pos3d(data["POS"]), data["RT3M"]),
        )
        * data["density"][0]
        * dx
    )
    MM = assemble(Mf)
    M = as_backend_type(MM).mat()
    Mass = PETSc.Mat()
    M.convert("dense", Mass)
    return Mass


def compute_stiffness(data):
    stress = Sigma(data, data["U"], data["UP"])
    stress_n = stress[:, 2]
    stress_1 = stress[:, 0]
    stress_2 = stress[:, 1]
    stress_s = as_tensor(
        [
            [stress_1[0], stress_2[0]],
            [stress_1[1], stress_2[1]],
            [stress_1[2], stress_2[2]],
        ]
    )

    ES = derivative(stress, data["U"], data["UT"])
    ES_t = derivative(stress_s, data["U"], data["UT"])
    ES_n = derivative(stress_s, data["UP"], data["UT"])
    En_t = derivative(stress_n, data["U"], data["UT"])
    En_n = derivative(stress_n, data["UP"], data["UT"])

    Mf = inner(data["UV"], En_n) * dx
    M = assemble(Mf)
    data["M"] = M

    Cf = inner(grad(data["UV"]), ES_n) * dx
    C = assemble(Cf)
    Hf = (inner(grad(data["UV"]), ES_n) - inner(data["UV"], En_t)) * dx
    H = assemble(Hf)
    data["H"] = H

    # the four initial solutions

    Escal = Constant(data["scaling_constraint"])
    Ef = inner(grad(data["UV"]), ES_t) * dx
    Ef += (
        (
            data["LV"][0] * data["UT"][0]
            + data["LV"][1] * data["UT"][1]
            + data["LV"][2] * data["UT"][2]
        )
        * Escal
        * dx
    )
    Ef += data["LV"][3] * dot(data["UT"], data["chains_d"][1][0]) * Escal * dx
    Ef += (
        (
            data["UV"][0] * data["LT"][0]
            + data["UV"][1] * data["LT"][1]
            + data["UV"][2] * data["LT"][2]
        )
        * Escal
        * dx
    )
    Ef += data["LT"][3] * dot(data["UV"], data["chains_d"][1][0]) * Escal * dx
    E = assemble(Ef)
    data["E"] = E
    S = (
        dot(stress_n, data["RV3F"]) * dx
        + dot(cross(pos3d(data["POS"]), stress_n), data["RV3M"]) * dx
    )
    L_res_f = derivative(S, data["UP"], data["UT"])
    data["L_res"] = assemble(L_res_f)
    R_res_f = derivative(S, data["U"], data["UT"])
    data["R_res"] = assemble(R_res_f)

    maxres = 0.0
    for i in range(4):
        tmp = E * data["chains"][i][0].vector()
        maxres = max(maxres, sqrt(tmp.inner(tmp)))
    for i in [2, 3]:
        tmp = -(H * data["chains"][i][0].vector()) - (E * data["chains"][i][1].vector())
        maxres = max(maxres, sqrt(tmp.inner(tmp)))

    #        if maxres > 1.E-16:
    #            scaling_factor = 1.E-16 / maxres;
    #        else:
    #            scaling_factor = 1.

    #        for i in range(4):
    #            data['chains'][i][0].vector()[:] = data['chains'][i][0].vector() * scaling_factor
    #        for i in [2, 3]:
    #            data['chains'][i][1].vector()[:] = data['chains'][i][1].vector() * scaling_factor
    for i in range(4):
        tmp = E * data["chains"][i][0].vector()
        maxres = max(maxres, sqrt(tmp.inner(tmp)))
    for i in [2, 3]:
        tmp = -(H * data["chains"][i][0].vector()) - (E * data["chains"][i][1].vector())
        maxres = max(maxres, sqrt(tmp.inner(tmp)))

    # solve E d1 = -H d0
    for i in range(2):
        rhs = -(H * data["chains"][i][0].vector())
        data["null_space"].orthogonalize(rhs)
        solve(E, data["chains"][i][1].vector(), rhs)
        data["null_space"].orthogonalize(data["chains"][i][1].vector())

    # solve E d2 = M d0 - H d1
    for i in [2, 3]:
        rhs = -(H * data["chains"][i][1].vector()) + (M * data["chains"][i][0].vector())
        data["null_space"].orthogonalize(rhs)
        solve(E, data["chains"][i][2].vector(), rhs)
        data["null_space"].orthogonalize(data["chains"][i][2].vector())

    a = np.zeros((2, 2))
    b = np.zeros((2, 1))
    for i in [2, 3]:
        res = -(H * data["chains"][i][2].vector()) + (M * data["chains"][i][1].vector())
        for k in range(2):
            b[k] = res.inner(data["chains"][k][0].vector())
            for ii in range(2):
                a[k, ii] = (
                    -(H * data["chains"][ii][1].vector())
                    + (M * data["chains"][ii][0].vector())
                ).inner(data["chains"][k][0].vector())
        x = np.linalg.solve(a, b)
        for ii in range(2):
            data["chains"][i][2].vector()[:] -= x[ii] * data["chains"][ii][1].vector()
            data["chains"][i][1].vector()[:] -= x[ii] * data["chains"][ii][0].vector()

    for i in [2, 3]:
        rhs = -(H * data["chains"][i][2].vector()) + (M * data["chains"][i][1].vector())
        data["null_space"].orthogonalize(rhs)
        solve(E, data["chains"][i][3].vector(), rhs)
        data["null_space"].orthogonalize(data["chains"][i][3].vector())

    # solve E d3 = M d1 - H d2
    for i in range(4):
        print("\nChain " + str(i) + ":")
        for k in range(len(data["chains"][i]) // 2, len(data["chains"][i])):
            print(
                "(d" + str(k) + ", d" + str(k) + ") = ",
                assemble(inner(data["chains_d"][i][k], data["chains_d"][i][k]) * dx),
            )
            print(
                "(l" + str(k) + ", l" + str(k) + ") = ",
                assemble(inner(data["chains_l"][i][k], data["chains_l"][i][k]) * dx),
            )
    for i in range(4):
        ll = len(data["chains"][i])
        for k in range(ll // 2, 0, -1):
            res = (
                E * data["chains"][i][ll - k].vector()
                + H * data["chains"][i][ll - 1 - k].vector()
            )
            if ll - 1 - k > 0:
                res -= M * data["chains"][i][ll - 2 - k].vector()
            res = as_backend_type(res).vec()
            print("residual chain", i, "order", ll, res.dot(res))
    print("")

    row1_col = []
    row2_col = []
    for i in range(6):
        row1_col.append(as_backend_type(data["chains"][0][0].vector().copy()).vec())
        row2_col.append(as_backend_type(data["chains"][0][0].vector().copy()).vec())

    M_p = as_backend_type(M).mat()
    C_p = as_backend_type(C).mat()
    E_p = as_backend_type(E).mat()
    S = PETSc.Mat().createDense([6, 6])
    S.setUp()

    B = PETSc.Mat().createDense([6, 6])
    B.setUp()

    G = PETSc.Mat().createDense([6, 6])
    G.setUp()

    g = PETSc.Vec().createMPI(6)

    Stiff = PETSc.Mat().createDense([6, 6])
    Stiff.setUp()

    col = -1
    for i in range(4):
        ll = len(data["chains"][i])
        for k in range(ll // 2, 0, -1):
            col = col + 1
            M_p.mult(
                as_backend_type(data["chains"][i][ll - 1 - k].vector()).vec(),
                row1_col[col],
            )
            C_p.multTransposeAdd(
                as_backend_type(data["chains"][i][ll - k].vector()).vec(),
                row1_col[col],
                row1_col[col],
            )
            C_p.mult(
                as_backend_type(data["chains"][i][ll - 1 - k].vector()).vec(),
                row2_col[col],
            )
            E_p.multAdd(
                as_backend_type(data["chains"][i][ll - k].vector()).vec(),
                row2_col[col],
                row2_col[col],
            )

    row = -1
    for i in range(4):
        ll = len(data["chains"][i])
        for k in range(ll // 2, 0, -1):
            row = row + 1
            for c in range(6):
                S.setValues(
                    row,
                    c,
                    as_backend_type(data["chains"][i][ll - 1 - k].vector())
                    .vec()
                    .dot(row1_col[c])
                    + as_backend_type(data["chains"][i][ll - k].vector())
                    .vec()
                    .dot(row2_col[c]),
                )
            B.setValues(
                row,
                range(6),
                as_backend_type(
                    data["L_res"] * data["chains"][i][ll - 1 - k].vector()
                    + data["R_res"] * data["chains"][i][ll - k].vector()
                ).vec(),
            )

    S.assemble()
    B.assemble()

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(S)
    ksp.setType(ksp.Type.PREONLY)  # Just use the preconditioner without a Krylov method
    pc = ksp.getPC()  # Preconditioner
    pc.setType(pc.Type.LU)  # Use a direct solve

    for i in range(6):
        ksp.solve(B.getColumnVector(i), g)
        G.setValues(range(6), i, g)

    G.assemble()

    G.transposeMatMult(S, B)
    B.matMult(G, Stiff)

    data["B"] = B
    data["G"] = G
    data["Stiff"] = Stiff
    return Stiff


def stress_field(data, force, moment, reference="local", voigt_convention="anba"):
    if reference == "local":
        stress_comp = lambda u, up: RotatedSigma(data, u, up)
    elif reference == "global":
        stress_comp = lambda u, up: Sigma(data, u, up)
    else:
        raise ValueError(
            'reference argument should be equal to either to"local" or to "global", got "'
            + reference
            + '" instead'
        )
    if voigt_convention == "anba":
        vector_conversion = stressTensorToStressVector
    elif voigt_convention == "paraview":
        vector_conversion = stressTensorToParaviewStressVector
    else:
        raise ValueError(
            'voigt_convention argument should be equal to either to"anba" or to "paraview", got "'
            + voigt_convention
            + '" instead'
        )

    eigensol_magnitudes = PETSc.Vec().createMPI(6)

    AzInt = PETSc.Vec().createMPI(6)

    AzInt.setValues(range(3), force)
    AzInt.setValues(range(3, 6), moment)
    AzInt.assemblyBegin()
    AzInt.assemblyEnd()

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(data["B"])
    ksp.setType(ksp.Type.PREONLY)  # Just use the preconditioner without a Krylov method
    pc = ksp.getPC()  # Preconditioner
    pc.setType(pc.Type.LU)  # Use a direct solve

    ksp.solve(AzInt, eigensol_magnitudes)

    UL = Function(data["UF3R4"])
    ULP = Function(data["UF3R4"])
    UL.vector()[:] = 0.0
    ULP.vector()[:] = 0.0
    row = -1
    for i in range(4):
        ll = len(data["chains"][i])
        for k in range(ll // 2, 0, -1):
            row = row + 1
            UL.vector()[:] += (
                data["chains"][i][ll - k].vector() * eigensol_magnitudes[row]
            )
            ULP.vector()[:] += (
                data["chains"][i][ll - 1 - k].vector() * eigensol_magnitudes[row]
            )
    (U, L) = split(UL)
    (UP, LP) = split(ULP)
    stress = local_project(vector_conversion(stress_comp(U, UP)), data["STRESS_FS"])
    stress.rename("stress tensor", "")
    return stress


def strain_field(data, force, moment, reference="local", voigt_convention="anba"):
    if reference == "local":
        strain_comp = lambda u, up: rotated_epsilon(data, u, up)
    elif reference == "global":
        strain_comp = lambda u, up: epsilon(u, up)
    else:
        raise ValueError(
            'reference argument should be equal to either to"local" or to "global", got "'
            + reference
            + '" instead'
        )
    if voigt_convention == "anba":
        vector_conversion = strainTensorToStrainVector
    elif voigt_convention == "paraview":
        vector_conversion = strainTensorToParaviewStrainVector
    else:
        raise ValueError(
            'voigt_convention argument should be equal to either to"anba" or to "paraview", got "'
            + voigt_convention
            + '" instead'
        )

    eigensol_magnitudes = PETSc.Vec().createMPI(6)

    AzInt = PETSc.Vec().createMPI(6)

    AzInt.setValues(range(3), force)
    AzInt.setValues(range(3, 6), moment)
    AzInt.assemblyBegin()
    AzInt.assemblyEnd()

    ksp = PETSc.KSP()
    ksp.create()
    ksp.setOperators(data["B"])
    ksp.setType(ksp.Type.PREONLY)  # Just use the preconditioner without a Krylov method
    pc = ksp.getPC()  # Preconditioner
    pc.setType(pc.Type.LU)  # Use a direct solve

    ksp.solve(AzInt, eigensol_magnitudes)

    UL = Function(data["UF3R4"])
    ULP = Function(data["UF3R4"])
    UL.vector()[:] = 0.0
    ULP.vector()[:] = 0.0
    row = -1
    for i in range(4):
        ll = len(data["chains"][i])
        for k in range(ll // 2, 0, -1):
            row = row + 1
            UL.vector()[:] += (
                data["chains"][i][ll - k].vector() * eigensol_magnitudes[row]
            )
            ULP.vector()[:] += (
                data["chains"][i][ll - 1 - k].vector() * eigensol_magnitudes[row]
            )
    (U, L) = split(UL)
    (UP, LP) = split(ULP)
    strain = local_project(vector_conversion(strain_comp(U, UP)), data["STRESS_FS"])
    strain.rename("strain tensor", "")
    return strain
