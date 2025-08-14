import numpy as np
from dolfin import UserExpression
import math


class Material:
    def __init__(self, rho=0.0):
        self.rho = rho
        self.transform_matrix = np.zeros((6, 6))
        self.mat_modulus = np.zeros((6, 6))
        self.mat_rotated_stress_modulus = np.zeros((6, 6))

    def Rho(self):
        return self.rho

    def transformation_matrix(self, alpha, beta):
        pi = math.pi
        pi180 = pi / 180.0

        sn_a = -math.sin(alpha * pi180)
        cn_a = math.cos(alpha * pi180)
        sn_b = -math.sin(beta * pi180)
        cn_b = math.cos(beta * pi180)

        self.transform_matrix[0, 0] = cn_a * cn_a * cn_b * cn_b
        self.transform_matrix[0, 1] = sn_a * sn_a
        self.transform_matrix[0, 2] = cn_a * cn_a * sn_b * sn_b
        self.transform_matrix[0, 3] = -2.0 * cn_a * sn_a * sn_b
        self.transform_matrix[0, 4] = -2.0 * cn_a * cn_a * sn_b * cn_b
        self.transform_matrix[0, 5] = 2.0 * cn_a * sn_a * cn_b

        self.transform_matrix[1, 0] = sn_a * sn_a * cn_b * cn_b
        self.transform_matrix[1, 1] = cn_a * cn_a
        self.transform_matrix[1, 2] = sn_a * sn_a * sn_b * sn_b
        self.transform_matrix[1, 3] = 2.0 * cn_a * sn_a * sn_b
        self.transform_matrix[1, 4] = -2.0 * sn_a * sn_a * sn_b * cn_b
        self.transform_matrix[1, 5] = -2.0 * cn_a * sn_a * cn_b

        self.transform_matrix[2, 0] = sn_b * sn_b
        self.transform_matrix[2, 2] = cn_b * cn_b
        self.transform_matrix[2, 4] = 2.0 * cn_b * sn_b

        self.transform_matrix[3, 0] = -sn_a * sn_b * cn_b
        self.transform_matrix[3, 2] = sn_a * sn_b * cn_b
        self.transform_matrix[3, 3] = cn_a * cn_b
        self.transform_matrix[3, 4] = -sn_a * cn_b * cn_b + sn_a * sn_b * sn_b
        self.transform_matrix[3, 5] = cn_a * sn_b

        self.transform_matrix[4, 0] = cn_a * sn_b * cn_b
        self.transform_matrix[4, 2] = -cn_a * sn_b * cn_b
        self.transform_matrix[4, 3] = sn_a * cn_b
        self.transform_matrix[4, 4] = -cn_a * sn_b * sn_b + cn_a * cn_b * cn_b
        self.transform_matrix[4, 5] = sn_a * sn_b

        self.transform_matrix[5, 0] = -sn_a * cn_a * cn_b * cn_b
        self.transform_matrix[5, 1] = cn_a * sn_a
        self.transform_matrix[5, 2] = -sn_a * cn_a * sn_b * sn_b
        self.transform_matrix[5, 3] = -cn_a * cn_a * sn_b + sn_a * sn_a * sn_b
        self.transform_matrix[5, 4] = 2.0 * sn_a * sn_b * cn_a * cn_b
        self.transform_matrix[5, 5] = cn_a * cn_a * cn_b - sn_a * sn_a * cn_b

        return self.transform_matrix

    def compute_elastic_modulus(self, alpha, beta):
        raise NotImplementedError

    def compute_rotated_stress_elastic_modulus(self, alpha, beta):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError


class IsotropicMaterial(Material):
    def __init__(self, mat_mechanic_prop, rho=0.0):
        super().__init__(rho)
        self.E = mat_mechanic_prop[0]
        self.nu = mat_mechanic_prop[1]
        E = self.E
        nu = self.nu
        G = E / (2 * (1 + nu))

        delta = E / (1.0 + nu) / (1 - 2.0 * nu)
        diag = (1.0 - nu) * delta
        off_diag = nu * delta

        self.mat_modulus[0, 0] = diag
        self.mat_modulus[0, 1] = off_diag
        self.mat_modulus[0, 2] = off_diag

        self.mat_modulus[1, 0] = off_diag
        self.mat_modulus[1, 1] = diag
        self.mat_modulus[1, 2] = off_diag

        self.mat_modulus[2, 0] = off_diag
        self.mat_modulus[2, 1] = off_diag
        self.mat_modulus[2, 2] = diag

        self.mat_modulus[3, 3] = G
        self.mat_modulus[4, 4] = G
        self.mat_modulus[5, 5] = G

    def compute_elastic_modulus(self, alpha, beta):
        return self.mat_modulus

    def compute_rotated_stress_elastic_modulus(self, alpha, beta):
        TM = self.transformation_matrix(alpha, beta)
        self.mat_rotated_stress_modulus = np.dot(self.mat_modulus, TM.T)
        return self.mat_rotated_stress_modulus

    def to_dict(self):
        return {
            "type": "isotropic",
            "E": self.E,
            "nu": self.nu,
            "rho": self.rho,
        }


class OrthotropicMaterial(Material):
    def __init__(self, mat_mechanic_prop, rho=0.0):
        super().__init__(rho)
        self.props = np.copy(mat_mechanic_prop)
        e_xx = mat_mechanic_prop[0, 0]
        e_yy = mat_mechanic_prop[0, 1]
        e_zz = mat_mechanic_prop[0, 2]
        g_yz = mat_mechanic_prop[1, 0]
        g_xz = mat_mechanic_prop[1, 1]
        g_xy = mat_mechanic_prop[1, 2]
        nu_zy = mat_mechanic_prop[2, 0]
        nu_zx = mat_mechanic_prop[2, 1]
        nu_xy = mat_mechanic_prop[2, 2]

        nu_yx = e_yy * nu_xy / e_xx
        nu_xz = e_xx * nu_zx / e_zz
        nu_yz = e_yy * nu_zy / e_zz

        self.mat_local_modulus = np.zeros((6, 6))

        delta = (
            1.0
            - nu_xy * nu_yx
            - nu_yz * nu_zy
            - nu_xz * nu_zx
            - 2.0 * nu_yx * nu_zy * nu_xz
        ) / (e_xx * e_yy * e_zz)
        self.mat_local_modulus[0, 0] = (1.0 - nu_yz * nu_zy) / (e_yy * e_zz * delta)
        self.mat_local_modulus[0, 1] = (nu_xy + nu_zy * nu_xz) / (e_xx * e_zz * delta)
        self.mat_local_modulus[0, 2] = (nu_xz + nu_xy * nu_yz) / (e_xx * e_yy * delta)

        self.mat_local_modulus[1, 0] = self.mat_local_modulus[0, 1]
        self.mat_local_modulus[1, 1] = (1 - nu_xz * nu_zx) / (e_xx * e_zz * delta)
        self.mat_local_modulus[1, 2] = (nu_yz + nu_yx * nu_xz) / (e_xx * e_yy * delta)

        self.mat_local_modulus[2, 0] = self.mat_local_modulus[0, 2]
        self.mat_local_modulus[2, 1] = self.mat_local_modulus[1, 2]
        self.mat_local_modulus[2, 2] = (1 - nu_xy * nu_yx) / (e_xx * e_yy * delta)

        self.mat_local_modulus[3, 3] = g_yz
        self.mat_local_modulus[4, 4] = g_xz
        self.mat_local_modulus[5, 5] = g_xy

    def compute_elastic_modulus(self, alpha, beta):
        TM = self.transformation_matrix(alpha, beta)
        self.mat_modulus = np.dot(np.dot(TM, self.mat_local_modulus), TM.T)
        return self.mat_modulus

    def compute_rotated_stress_elastic_modulus(self, alpha, beta):
        TM = self.transformation_matrix(alpha, beta)
        self.mat_rotated_stress_modulus = np.dot(self.mat_local_modulus, TM.T)
        return self.mat_rotated_stress_modulus

    def to_dict(self):
        return {
            "type": "orthotropic",
            "props": self.props.tolist(),
            "rho": self.rho,
        }


class ElasticModulus(UserExpression):
    def __init__(
        self, mats_library, material_id, plane_orientation, fiber_orientation, **kwargs
    ):
        super().__init__(**kwargs)
        self.mats_library = mats_library
        self.material_id = material_id
        self.plane_orientation = plane_orientation
        self.fiber_orientation = fiber_orientation

    def eval_cell(self, values, x, cell):
        mat_id = self.material_id[cell.index]
        alpha = self.plane_orientation[cell.index]
        beta = self.fiber_orientation[cell.index]
        transformed_stiffness = self.mats_library[mat_id].compute_elastic_modulus(
            alpha, beta
        )
        values[:] = transformed_stiffness.flatten()

    def value_shape(self):
        return (36,)


class RotatedStressElasticModulus(UserExpression):
    def __init__(
        self, mats_library, material_id, plane_orientation, fiber_orientation, **kwargs
    ):
        super().__init__(**kwargs)
        self.mats_library = mats_library
        self.material_id = material_id
        self.plane_orientation = plane_orientation
        self.fiber_orientation = fiber_orientation

    def eval_cell(self, values, x, cell):
        mat_id = self.material_id[cell.index]
        alpha = self.plane_orientation[cell.index]
        beta = self.fiber_orientation[cell.index]
        transformed_stiffness = self.mats_library[
            mat_id
        ].compute_rotated_stress_elastic_modulus(alpha, beta)
        values[:] = transformed_stiffness.flatten()

    def value_shape(self):
        return (36,)


class TransformationMatrix(UserExpression):
    def __init__(
        self, mats_library, material_id, plane_orientation, fiber_orientation, **kwargs
    ):
        super().__init__(**kwargs)
        self.mats_library = mats_library
        self.material_id = material_id
        self.plane_orientation = plane_orientation
        self.fiber_orientation = fiber_orientation

    def eval_cell(self, values, x, cell):
        mat_id = self.material_id[cell.index]
        alpha = self.plane_orientation[cell.index]
        beta = self.fiber_orientation[cell.index]
        rot_matrix = self.mats_library[mat_id].transformation_matrix(alpha, beta)
        values[:] = rot_matrix.flatten()

    def value_shape(self):
        return (36,)


class MaterialDensity(UserExpression):
    def __init__(self, mats_library, material_id, **kwargs):
        super().__init__(**kwargs)
        self.mats_library = mats_library
        self.material_id = material_id

    def eval_cell(self, values, x, cell):
        mat_id = self.material_id[cell.index]
        values[0] = self.mats_library[mat_id].Rho()

    def value_shape(self):
        return (1,)
