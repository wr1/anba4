import argparse
import json
import dolfin
import os
import multiprocessing as mp
from typing import List
from .. import (
    initialize_anba_model,
    initialize_fe_functions,
    initialize_chains,
    compute_stiffness,
    compute_inertia,
    stress_field,
    strain_field,
    ComputeShearCenter,
    ComputeTensionCenter,
    ComputeMassCenter,
    DecoupleStiffness,
    PrincipalAxesRotationAngle,
)
from ..io.export import (
    import_model_json,
    serialize_matrix,
    serialize_field,
    serialize_numpy_matrix,
)


# Set shared cache dir early
os.environ['DIJITSO_CACHE_DIR'] = os.path.join(os.getcwd(), 'cache')


def run_single_calculation(
    input_path: str,
    output_path: str,
    force: List[float],
    moment: List[float],
    reference: str,
    voigt: str,
    save_fields: str,
):
    """Run a single ANBA4 calculation."""
    # Print out to diagnose problem
    print("Dolfin version:", getattr(dolfin, '__version__', 'unknown'))
    print("Dolfin attributes with 'thread':", [attr for attr in dir(dolfin) if 'thread' in attr.lower()])
    print("Dolfin parameters keys:", list(dolfin.parameters.keys()) if hasattr(dolfin.parameters, 'keys') else 'no keys method')
    # Disable parallel processing to prevent caching issues
    os.environ["OMP_NUM_THREADS"] = "1"
    dolfin.set_log_level(dolfin.LogLevel.WARNING)

    serializable_input = import_model_json(input_path)
    anbax_data = initialize_anba_model(serializable_input)
    initialize_fe_functions(anbax_data)
    initialize_chains(anbax_data)
    stiff = compute_stiffness(anbax_data)
    mass = compute_inertia(anbax_data)
    stress = stress_field(anbax_data, force, moment, reference, voigt)
    strain = strain_field(anbax_data, force, moment, reference, voigt)

    # Compute centers and angles
    shear_center = ComputeShearCenter(stiff)
    tension_center = ComputeTensionCenter(stiff)
    mass_center = ComputeMassCenter(mass)
    decoupled_stiff = DecoupleStiffness(stiff)
    principal_angle = PrincipalAxesRotationAngle(decoupled_stiff)

    output_data = {
        "stiffness": serialize_matrix(stiff),
        "mass": serialize_matrix(mass),
        "stress": serialize_field(stress),
        "strain": serialize_field(strain),
        "shear_center": shear_center,
        "tension_center": tension_center,
        "mass_center": mass_center,
        "decoupled_stiffness": serialize_numpy_matrix(decoupled_stiff),
        "principal_angle": principal_angle,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    if save_fields:
        result_file = dolfin.XDMFFile(save_fields)
        result_file.parameters["functions_share_mesh"] = True
        result_file.parameters["rewrite_function_mesh"] = False
        result_file.parameters["flush_output"] = True
        result_file.write(stress, t=0.0)
        result_file.write(strain, t=1.0)
        print(f"Fields saved to {save_fields}")

    print(f"Outputs serialized to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(description="CLI tool to run ANBA4 computations from JSON input and serialize outputs to JSON.")
    parser.add_argument('-i', '--inputs', type=str, nargs='+', required=True, help='Input JSON files (one for single run, multiple for batch)')
    parser.add_argument('--output-post', type=str, default='_out', help='Postfix for output files (e.g., _out for file_out.json)')
    parser.add_argument('--force', type=float, nargs=3, default=[1.0, 0.0, 0.0], help='Force vector (fx fy fz)')
    parser.add_argument('--moment', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Moment vector (mx my mz)')
    parser.add_argument('--reference', type=str, default='local', choices=['local', 'global'], help='Reference system for fields (local or global)')
    parser.add_argument('-v', '--voigt', type=str, default='anba', choices=['anba', 'paraview'], help='Voigt convention (anba or paraview)')
    parser.add_argument('-s', '--save-fields', type=str, default=None, help='Base XDMF file to save stress and strain fields (appends postfix)')

    args = parser.parse_args()

    inputs = args.inputs
    num_runs = len(inputs)

    # Prepare output paths and save fields
    output_paths = []
    save_fields_list = []
    for inp in inputs:
        base = inp.replace('.json', '')
        output_paths.append(f"{base}{args.output_post}.json")
        if args.save_fields:
            save_fields_list.append(f"{base}{args.output_post}.xdmf")
        else:
            save_fields_list.append(None)

    if num_runs == 1:
        # Single run
        run_single_calculation(inputs[0], output_paths[0], args.force, args.moment, args.reference, args.voigt, save_fields_list[0])
    else:
        # Batch processing: precompile JIT serially with first input, then run all in parallel
        print("Pre-compiling JIT...")
        temp_output = f"{inputs[0].replace('.json', '')}_temp.json"
        run_single_calculation(inputs[0], temp_output, args.force, args.moment, args.reference, args.voigt, None)
        os.remove(temp_output)  # Remove temp file

        # Prepare params for parallel runs
        params = list(zip(inputs, output_paths, [args.force]*num_runs, [args.moment]*num_runs, [args.reference]*num_runs, [args.voigt]*num_runs, save_fields_list))

        # Use 'spawn' start method for clean processes
        mp.set_start_method('spawn', force=True)

        with mp.Pool(processes=min(mp.cpu_count(), len(params))) as pool:
            results = pool.starmap(run_single_calculation, params)

        print(f"Batch processing complete. Outputs saved next to inputs with postfix {args.output_post}")


if __name__ == '__main__':
    main()
