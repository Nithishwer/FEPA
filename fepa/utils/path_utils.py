"""This module contains functions to load information from a config file"""

import os


def load_paths_for_compound(config: dict, cmp: str, bp_selection_string: str) -> dict:
    """Loads MD trajectory paths for a given compound."""
    path_dict = {}
    for van_rep_no in [1, 2, 3]:
        path_dict[f"{cmp}_van_{van_rep_no}"] = {
            "pdb": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "npt.gro",
            ),
            "xtc": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "prod.xtc",
            ),
            "bp_selection_string": bp_selection_string,
        }
        path_dict[f"{cmp}_nvt"] = {
            "pdb": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "nvt.gro",
            ),
            "xtc": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "nvt.xtc",
            ),
            "bp_selection_string": bp_selection_string,
        }
    for van_rep_no in [1, 2, 3]:
        path_dict[f"apo_{van_rep_no}"] = {
            "pdb": os.path.join(
                config["apo_path_template"].format(REP_NO=van_rep_no), "npt.gro"
            ),
            "xtc": os.path.join(
                config["apo_path_template"].format(REP_NO=van_rep_no), "prod.xtc"
            ),
            "bp_selection_string": bp_selection_string,
        }
    return path_dict


def load_abfe_paths_for_compound(
    config: dict,
    cmp: str,
    bp_selection_string: str,
    van_list: list[str],
    leg_window_list: list[str],
) -> dict:
    """Loads MD trajectory paths for a given compound."""
    path_dict = {}
    for van_rep_no in van_list:
        for leg_window in leg_window_list:
            path_dict[f"{cmp}_van_{van_rep_no}_{leg_window}"] = {
                "pdb": os.path.join(
                    config["abfe_window_path_template"].format(
                        CMP_NAME=cmp,
                        REP_NO=van_rep_no,
                        STAGE="npt_pr",
                        ABFE_REP_NO=1,
                        LEG_WINDOW=leg_window,
                    ),
                    "confout.gro",
                ),
                "xtc": os.path.join(
                    config["abfe_window_path_template"].format(
                        CMP_NAME=cmp,
                        REP_NO=van_rep_no,
                        STAGE="prod",
                        ABFE_REP_NO=1,
                        LEG_WINDOW=leg_window,
                    ),
                    "traj_comp.xtc",
                ),
                "bp_selection_string": bp_selection_string,
            }
            path_dict[f"{cmp}_nvt"] = {
                "pdb": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "nvt.gro",
                ),
                "xtc": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "nvt.xtc",
                ),
                "bp_selection_string": bp_selection_string,
            }
            path_dict[f"{cmp}_van_{van_rep_no}"] = {
                "pdb": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "npt.gro",
                ),
                "xtc": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "prod.xtc",
                ),
                "bp_selection_string": bp_selection_string,
            }
    for van_rep_no in [1, 2, 3]:
        path_dict[f"apo_{van_rep_no}"] = {
            "pdb": os.path.join(
                config["apo_path_template"].format(REP_NO=van_rep_no), "npt.gro"
            ),
            "xtc": os.path.join(
                config["apo_path_template"].format(REP_NO=van_rep_no), "prod.xtc"
            ),
            "bp_selection_string": bp_selection_string,
        }
    return path_dict


def load_paths_for_apo(config: dict, bp_selection_string: str) -> dict:
    """Loads apo MD trajectory paths from config file"""
    path_dict = {}
    for apo_rep_no in [1, 2, 3]:
        path_dict[f"apo_{apo_rep_no}"] = {
            "pdb": os.path.join(
                config["apo_path_template"].format(REP_NO=apo_rep_no), "npt.gro"
            ),
            "xtc": os.path.join(
                config["apo_path_template"].format(REP_NO=apo_rep_no), "prod.xtc"
            ),
            "bp_selection_string": bp_selection_string,
        }
    return path_dict


def load_paths_for_memento_equil(
    sim_path_template: str, bp_selection_string: str
) -> dict:
    """Loads MD trajectory paths for memento equil from config file"""
    path_dict = {}
    for window in range(24):
        sim_folder = f"sim{window}"
        path_dict[sim_folder] = {
            "pdb": os.path.join(
                sim_path_template.format(SIM_FOLDER=sim_folder), "npt.gro"
            ),
            "xtc": os.path.join(
                sim_path_template.format(SIM_FOLDER=sim_folder), "prod.xtc"
            ),
            "bp_selection_string": bp_selection_string,
        }
    return path_dict
