import json
import numpy as np
import pyvista as pv
import dolfin

from ..material.material_py import Material


def export_model_vtu(
    mesh: dolfin.Mesh,
    materials: dolfin.MeshFunction,
    fiber_orientations: dolfin.MeshFunction,
    plane_orientations: dolfin.MeshFunction,
    mesh_name: str = "mesh.vtu",
):
    """Export model to VTU format using PyVista."""
    pts = np.hstack((mesh.coordinates(), np.zeros((mesh.coordinates().shape[0], 1))))

    cells = np.hstack(
        (3 * np.ones((mesh.cells().shape[0], 1)).astype(np.int64), mesh.cells())
    )

    grd = pv.UnstructuredGrid(
        cells,
        [pv.CellType.TRIANGLE for i in range(mesh.cells().shape[0])],
        pts,
    )
    grd.cell_data["Materials"] = materials.array()
    grd.cell_data["FiberOrientations"] = fiber_orientations.array()
    grd.cell_data["PlaneOrientations"] = plane_orientations.array()
    grd.save(mesh_name)


def export_model_json(
    mesh: dolfin.Mesh,
    mat_library: list[Material],
    materials: dolfin.MeshFunction,
    fiber_orientations: dolfin.MeshFunction,
    plane_orientations: dolfin.MeshFunction,
    degree: int,
    scaling_constraint: float = 1.0,
    singular: bool = False,
    filename: str = "model.json",
):
    """Export model input data to JSON format, including full material properties."""
    points = [list(p) + [0.0] for p in mesh.coordinates().tolist()]
    cells = mesh.cells().tolist()
    mat_ids = materials.array().tolist()
    fiber_or = fiber_orientations.array().tolist()
    plane_or = plane_orientations.array().tolist()
    mat_lib = [m.to_dict() for m in mat_library]
    data = {
        "points": points,
        "cells": cells,
        "material_ids": mat_ids,
        "fiber_orientations": fiber_or,
        "plane_orientations": plane_or,
        "mat_library": mat_lib,
        "degree": degree,
        "scaling_constraint": scaling_constraint,
        "singular": singular,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
