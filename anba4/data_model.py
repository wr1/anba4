from pydantic import BaseModel
from typing import Any, List, Optional
from dolfin import *


class AnbaData(BaseModel):
    mesh: Any
    degree: int
    matLibrary: List[Any]
    materials: Any
    fiber_orientations: Any
    plane_orientations: Any
    scaling_constraint: float = 1.0
    modulus: Any
    RotatedStress_modulus: Any
    MaterialRotation_matrix: Any
    density: Any
    POS: Any
    UF3: Optional[Any] = None
    R3: Optional[Any] = None
    R3R3: Optional[Any] = None
    RV3F: Optional[Any] = None
    RV3M: Optional[Any] = None
    RT3F: Optional[Any] = None
    RT3M: Optional[Any] = None
    STRESS_FS: Optional[Any] = None
    R4: Optional[Any] = None
    UF3R4: Optional[Any] = None
    UL: Optional[Any] = None
    U: Optional[Any] = None
    L: Optional[Any] = None
    ULP: Optional[Any] = None
    UP: Optional[Any] = None
    LP: Optional[Any] = None
    ULV: Optional[Any] = None
    UV: Optional[Any] = None
    LV: Optional[Any] = None
    ULT: Optional[Any] = None
    UT: Optional[Any] = None
    LT: Optional[Any] = None
    base_chains_expression: Optional[List[Any]] = None
    linear_chains_expression: Optional[List[Any]] = None
    Torsion: Optional[Any] = None
    Flex_y: Optional[Any] = None
    Flex_x: Optional[Any] = None
    chains: Optional[List[List[Any]]] = None
    chains_d: Optional[List[List[Any]]] = None
    chains_l: Optional[List[List[Any]]] = None
    null_space: Optional[Any] = None
    M: Optional[Any] = None
    H: Optional[Any] = None
    E: Optional[Any] = None
    L_res: Optional[Any] = None
    R_res: Optional[Any] = None
    B: Optional[Any] = None
    G: Optional[Any] = None
    Stiff: Optional[Any] = None
