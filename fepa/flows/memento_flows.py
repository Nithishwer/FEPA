import os
import shutil
import logging
import gromacs
from fepa.utils.memento_utils import (
    prepare_memento_input,
    run_pymemento,
)


class memento_workflow:
    def __init__(
        self,
        memento_dir,
        initial_gro,
        target_gro,
        initial_name,
        target_name,
        template_path,
        protonation_states=None,
        run_name="memento_run_v1",
    ):
        self.memento_dir = memento_dir
        self.initial_gro = initial_gro
        self.target_gro = target_gro
        self.initial_name = initial_name
        self.target_name = target_name
        self.template_path = template_path
        self.protonation_states = protonation_states
        self.run_name = run_name
        self.folder_name = f"{self.initial_name}_{self.target_name}"
        self.folder_path = os.path.join(self.memento_dir, self.folder_name)

    def prepare_memento(self):
        logging.info(
            f"Preparing memento input for {self.initial_name} and {self.target_name}."
        )
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"Folder '{self.folder_name}' created successfully.")
            os.chdir(self.folder_path)
        else:
            print(f"Folder '{self.folder_name}' already exists.")
            os.chdir(self.folder_path)

        prepare_memento_input(
            initial_gro=self.initial_gro,
            target_gro=self.target_gro,
            run_name=self.run_name,
        )

    def run_memento(self, template_path, last_run, protonation_states=None):
        if not os.path.exists(os.path.join(self.folder_path, self.run_name)):
            print(
                f"Folder '{os.path.join(self.folder_path, self.run_name)}' does not exist."
            )
            print("Prepare memento input first.")
            exit(1)
        else:
            print(f"Folder '{os.path.join(self.folder_path, self.run_name)}' exists.")
            os.chdir(os.path.join(self.folder_path, self.run_name))

        run_pymemento(
            template_path=template_path,
            # template_path="/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1_memento/apo_template",
            last_run=last_run,
            protonation_states=protonation_states,
        )
        os.chdir("../..")

    def prepare_equil_simulations(
        self, job_script_template, index_selection="1|13|14|15"
    ):
        # Navigate to memento_dir/wdir/boxes
        print(f"Setting up equilibration for the complex in {self.memento_dir}.")
        boxes_dir = os.path.join(self.folder_path, self.run_name, "wdir", "boxes")

        # Iterate through folders starting with 'sim'
        for folder_name in os.listdir(boxes_dir):
            if folder_name.startswith("sim"):
                folder_path = os.path.join(boxes_dir, folder_name)
                current_dir = os.getcwd()
                os.chdir(folder_path)
                gromacs.make_ndx(
                    f="em.gro", o="index.ndx", input=(index_selection, "q")
                )
                os.chdir(current_dir)

        # Copy the job script template
        job_script_dest = os.path.join(
            self.folder_path, self.run_name, "wdir", "boxes", "job_equil_ranv_arr.sh"
        )
        shutil.copy(job_script_template, job_script_dest)

        # Replace all occurrences of JOBNAME with the folder one level up from self.memento_dir
        jobname = self.initial_name + "_" + self.target_name
        with open(job_script_dest, "r") as file:
            script_content = file.read()

        # Replace JOBNAME in the script content
        script_content = script_content.replace("JOBNAME", jobname)

        # Write the modified script back to the file
        logging.info(f"Writing the modified script to {job_script_dest}.")
        with open(job_script_dest, "w") as file:
            file.write(script_content)

        print(f"Equilibration setup completed for {self.memento_dir}.")
