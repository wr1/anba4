from pydantic import BaseModel
from typing import Any, List, Optional
from dolfin import *

class Material(BaseModel):
    name: str
    E: float  # Young's modulus
    nu: float  # Poisson's ratio
    density: float  # Density of the material
    fiber_orientation: List[float]  # Fiber orientation in Voigt notation
    plane_orientation: List[float]  # Plane orientation in Voigt notation
    scaling_constraint: float = 1.0  # Scaling constraint for the material

class MaterialData(BaseModel):
    modulus: Optional[Any] = None
    RotatedStress_modulus: Optional[Any] = None
    MaterialRotation_matrix: Optional[Any] = None
    density: Optional[Any] = None

class InputData(BaseModel):
    mesh: Any
    degree: int
    matLibrary: Optional[List[Any]] = None
    materials: Any
    fiber_orientations: Any
    plane_orientations: Any
    scaling_constraint: float = 1.0
    singular: bool = False

class FEFunctions(BaseModel):
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

class Chains(BaseModel):
    base_chains_expression: Optional[List[Any]] = None
    linear_chains_expression: Optional[List[Any]] = None
    Torsion: Optional[Any] = None
    Flex_y: Optional[Any] = None
    Flex_x: Optional[Any] = None
    chains: Optional[List[List[Any]]] = None
    chains_d: Optional[List[List[Any]]] = None
    chains_l: Optional[List[List[Any]]] = None

class OutputData(BaseModel):
    null_space: Optional[Any] = None
    M: Optional[Any] = None
    H: Optional[Any] = None
    E: Optional[Any] = None
    L_res: Optional[Any] = None
    R_res: Optional[Any] = None
    B: Optional[Any] = None
    G: Optional[Any] = None
    Stiff: Optional[Any] = None

class AnbaData(BaseModel):
    input_data: InputData
    material_data: MaterialData = MaterialData()
    fe_functions: FEFunctions
    chains: Chains
    output_data: OutputData

