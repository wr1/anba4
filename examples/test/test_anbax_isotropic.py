import unittest
import numpy as np
from dolfin import *
from anba4 import *


class Test_Isotropic(unittest.TestCase):
    def testStiffMass(self):
        parameters["form_compiler"]["optimize"] = True
        parameters["form_compiler"]["quadrature_degree"] = 2

        # Basic material parameters. 9 is needed for orthotropic materials.

        E = 1.0
        nu = 0.33
        # Assmble into material mechanical property Matrix.
        matMechanicProp = [E, nu]
        # Meshing domain.

        mesh = UnitSquareMesh(10, 10)
        ALE.move(mesh, Constant([-0.5, -0.5]))

        # CompiledSubDomain
        materials = MeshFunction("size_t", mesh, mesh.topology().dim())
        fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
        plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())

        materials.set_all(0)
        fiber_orientations.set_all(0.0)
        plane_orientations.set_all(90.0)

        # Build material property library.
        mat1 = material.IsotropicMaterial(matMechanicProp, 1.0)

        matLibrary = []
        matLibrary.append(mat1)

        input_data = InputData(
            mesh=mesh,
            degree=2,
            matLibrary=matLibrary,
            materials=materials,
            plane_orientations=plane_orientations,
            fiber_orientations=fiber_orientations,
        )
        anbax_data = initialize_anba_model(input_data)
        initialize_fe_functions(anbax_data)
        initialize_chains(anbax_data)
        stiff = compute_stiffness(anbax_data)
        stiff.view()

        mass = compute_inertia(anbax_data)
        mass.view()

        stress_result_file = XDMFFile("Stress.xdmf")
        stress_result_file.parameters["functions_share_mesh"] = True
        stress_result_file.parameters["rewrite_function_mesh"] = False
        stress_result_file.parameters["flush_output"] = True

        stress = stress_field(
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

        np.testing.assert_almost_equal(
            stiff.getValues(range(6), range(6)),
            np.array(
                [
                    [
                        3.11064401e-01,
                        -5.76267647e-07,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        2.19833734e-16,
                    ],
                    [
                        -5.76267647e-07,
                        3.11064401e-01,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        -2.66259961e-16,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        1.00000000e00,
                        5.13607660e-16,
                        -3.11112792e-16,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        4.93713695e-16,
                        8.33333333e-02,
                        -8.92869184e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        -2.32228751e-16,
                        -8.22353985e-17,
                        8.33333333e-02,
                        0.00000000e00,
                    ],
                    [
                        2.14650429e-16,
                        -2.41327387e-16,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        5.28559454e-02,
                    ],
                ]
            ),
            6,
        )

        np.testing.assert_almost_equal(
            mass.getValues(range(6), range(6)),
            np.array(
                [
                    [
                        1.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        3.64291930e-17,
                    ],
                    [
                        0.00000000e00,
                        1.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        -1.30104261e-17,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        1.00000000e00,
                        -3.64291930e-17,
                        1.30104261e-17,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        -3.64291930e-17,
                        8.33333333e-02,
                        -2.60208521e-18,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        1.30104261e-17,
                        -2.60208521e-18,
                        8.33333333e-02,
                        0.00000000e00,
                    ],
                    [
                        3.64291930e-17,
                        -1.30104261e-17,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        1.66666667e-01,
                    ],
                ]
            ),
            6,
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test_Isotropic))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
