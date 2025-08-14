import click
import json
import dolfin
from typing import List
from .. import (
    initialize_anba_model,
    initialize_fe_functions,
    initialize_chains,
    compute_stiffness,
    compute_inertia,
    stress_field,
    strain_field,
)
from ..io.export import import_model_json, serialize_matrix, serialize_field


@click.command()
@click.option(
    "--input", "-i", type=click.Path(exists=True), required=True, help="Input JSON file"
)
@click.option(
    "--output", "-o", type=click.Path(), default="output.json", help="Output JSON file"
)
@click.option(
    "--force",
    type=(float, float, float),
    default=(1.0, 0.0, 0.0),
    help="Force vector (fx fy fz)",
)
@click.option(
    "--moment",
    type=(float, float, float),
    default=(0.0, 0.0, 0.0),
    help="Moment vector (mx my mz)",
)
@click.option(
    "--reference",
    type=str,
    default="local",
    help="Reference system for fields (local or global)",
)
@click.option(
    "--voigt", type=str, default="anba", help="Voigt convention (anba or paraview)"
)
@click.option(
    "--save-fields",
    type=click.Path(),
    default=None,
    help="XDMF file to save stress and strain fields",
)
def run(
    input: str,
    output: str,
    force: List[float],
    moment: List[float],
    reference: str,
    voigt: str,
    save_fields: str,
):
    """CLI tool to run ANBA4 computations from JSON input and serialize outputs to JSON."""
    serializable_input = import_model_json(input)
    anbax_data = initialize_anba_model(serializable_input)
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    mass = compute_inertia(anbax_data)
    stress = stress_field(anbax_data, force, moment, reference, voigt)
    strain = strain_field(anbax_data, force, moment, reference, voigt)

    output_data = {
        "stiffness": serialize_matrix(stiff),
        "mass": serialize_matrix(mass),
        "stress": serialize_field(stress),
        "strain": serialize_field(strain),
    }

    with open(output, "w") as f:
        json.dump(output_data, f, indent=4)

    if save_fields:
        result_file = dolfin.XDMFFile(save_fields)
        result_file.parameters["functions_share_mesh"] = True
        result_file.parameters["rewrite_function_mesh"] = False
        result_file.parameters["flush_output"] = True
        result_file.write(stress, t=0.0)
        result_file.write(strain, t=1.0)
        print(f"Fields saved to {save_fields}")

    print(f"Outputs serialized to {output}")
