"""Orthotropic material model."""

import numpy as np
from .base import Material


class OrthotropicMaterial(Material):
    """Orthotropic material."""

    def __init__(self, E, G, nu, rho=0.0):
        super().__init__(rho)
        self.E = np.array(E)
        self.G = np.array(G)
        self.nu = np.array(nu)
        e_xx, e_yy, e_zz = self.E
        g_yz, g_xz, g_xy = self.G
        nu_zy, nu_zx, nu_xy = self.nu

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
        """Compute elastic modulus."""
        TM = self.transformation_matrix(alpha, beta)
        self.mat_modulus = np.dot(np.dot(TM, self.mat_local_modulus), TM.T)
        return self.mat_modulus

    def compute_rotated_stress_elastic_modulus(self, alpha, beta):
        """Compute rotated stress elastic modulus."""
        TM = self.transformation_matrix(alpha, beta)
        self.mat_rotated_stress_modulus = np.dot(self.mat_local_modulus, TM.T)
        return self.mat_rotated_stress_modulus

    def to_dict(self):
        """Serialize to dict."""
        return {
            "type": "orthotropic",
            "E": self.E.tolist(),
            "G": self.G.tolist(),
            "nu": self.nu.tolist(),
            "rho": self.rho,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Deserialize from dict."""
        return cls(
            np.array(d["E"]),
            np.array(d["G"]),
            np.array(d["nu"]),
            d["rho"],
        )
