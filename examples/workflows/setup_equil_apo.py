import os
import shutil
import gromacs
import logging

print(gromacs.release())


def setup_equil_complex(memento_dir):
    # Navigate to memento_dir/wdir/boxes
    print(f"Setting up equilibration for the complex in {memento_dir}.")
    boxes_dir = os.path.join(memento_dir, "wdir", "boxes")

    # Iterate through folders starting with 'sim'
    for folder_name in os.listdir(boxes_dir):
        if folder_name.startswith("sim"):
            folder_path = os.path.join(boxes_dir, folder_name)
            current_dir = os.getcwd()
            os.chdir(folder_path)
            gromacs.make_ndx(f="em.gro", o="index.ndx", input=("1|13|14|15", "q"))
            os.chdir(current_dir)

    # Copy the job script template
    job_script_source = "/biggin/b211/reub0138/Projects/orexin/lenselink_a2a_memento_v1/job_vanilla_soma_equil_arr_template.sh"
    job_script_dest = os.path.join(
        memento_dir, "wdir", "boxes", "job_equil_ranv_arr.sh"
    )
    shutil.copy(job_script_source, job_script_dest)

    # Replace all occurrences of JOBNAME with the folder one level up from memento_dir
    jobname = os.path.basename(os.path.dirname(memento_dir))
    with open(job_script_dest, "r") as file:
        script_content = file.read()

    # Replace JOBNAME in the script content
    script_content = script_content.replace("JOBNAME", jobname)

    # Write the modified script back to the file
    logging.info(f"Writing the modified script to {job_script_dest}.")
    with open(job_script_dest, "w") as file:
        file.write(script_content)

    print(f"Equilibration setup completed for {memento_dir}.")


def main():
    # Example: You can replace the input path with a dynamic input mechanism if needed.
    apo_pairs = [("apo_1", "apo_2"), ("apo_1", "apo_3"), ("apo_2", "apo_3")]
    memento_run_name = "memento_run_v1"
    for apo_pair in apo_pairs:
        memento_dir = f"/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1_memento/{apo_pair[0]}_{apo_pair[1]}/{memento_run_name}"
        setup_equil_complex(memento_dir)


# Ensure the script runs only when executed as the main program
if __name__ == "__main__":
    main()
