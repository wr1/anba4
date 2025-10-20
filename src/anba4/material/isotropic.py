"""Isotropic material model."""

import numpy as np
from .base import Material


class IsotropicMaterial(Material):
    """Isotropic material."""

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
        """Compute elastic modulus."""
        return self.mat_modulus

    def compute_rotated_stress_elastic_modulus(self, alpha, beta):
        """Compute rotated stress elastic modulus."""
        TM = self.transformation_matrix(alpha, beta)
        self.mat_rotated_stress_modulus = np.dot(self.mat_modulus, TM.T)
        return self.mat_rotated_stress_modulus

    def to_dict(self):
        """Serialize to dict."""
        return {
            "type": "isotropic",
            "E": self.E,
            "nu": self.nu,
            "rho": self.rho,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Deserialize from dict."""
        mat_mechanic_prop = [d["E"], d["nu"]]
        return cls(mat_mechanic_prop, d["rho"])
