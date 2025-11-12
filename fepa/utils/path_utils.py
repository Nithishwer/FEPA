"""This module contains functions to load information from a config file"""

import os


def load_paths_for_compound(
    config: dict,
    cmp: str,
    bp_selection_string: str,
    apo=True,
    vanilla_path_template_old=False,
) -> dict:
    """Loads MD trajectory paths for a given compound."""
    path_dict = {}
    for van_rep_no in [1, 2, 3]:
        path_dict[f"{cmp}_van_{van_rep_no}"] = {
            "pdb": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1 and vanilla_path_template_old
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "npt.gro",
            ),
            "xtc": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1 and vanilla_path_template_old
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "prod.xtc",
            ),
            "tpr": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1 and vanilla_path_template_old
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "prod.tpr",
            ),
            "bp_selection_string": bp_selection_string,
        }
        path_dict[f"{cmp}_nvt"] = {
            "pdb": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1 and vanilla_path_template_old
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "nvt.gro",
            ),
            "xtc": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1 and vanilla_path_template_old
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "nvt.xtc",
            ),
            "tpr": os.path.join(
                config[
                    "vanilla_path_template_old"
                    if van_rep_no == 1 and vanilla_path_template_old
                    else "vanilla_path_template"
                ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                "nvt.tpr",
            ),
            "bp_selection_string": bp_selection_string,
        }
    if apo:
        for van_rep_no in [1, 2, 3]:
            path_dict[f"apo_{van_rep_no}"] = {
                "pdb": os.path.join(
                    config["apo_path_template"].format(REP_NO=van_rep_no), "npt.gro"
                ),
                "xtc": os.path.join(
                    config["apo_path_template"].format(REP_NO=van_rep_no), "prod.xtc"
                ),
                "tpr": os.path.join(
                    config["apo_path_template"].format(REP_NO=van_rep_no), "prod.tpr"
                ),
                "bp_selection_string": bp_selection_string,
            }
    return path_dict


def load_paths_for_memento_boxes(
    boxes_path: str,
    nboxes: int,
    bp_selection_string: str,
    name: str,
) -> dict:
    """Loads MD trajectory paths for a given boxes path"""
    path_dict = {}
    for i in range(nboxes):
        path_dict[f"{name}_{i}"] = {
            "pdb": os.path.join(os.path.join(boxes_path, f"sim{i}"), "npt.gro"),
            "xtc": os.path.join(os.path.join(boxes_path, f"sim{i}"), "prod.xtc"),
            "tpr": os.path.join(os.path.join(boxes_path, f"sim{i}"), "prod.tpr"),
            "bp_selection_string": bp_selection_string,
        }
    return path_dict


def load_paths_for_reus_boxes(
    boxes_path: str,
    nboxes: int,
    bp_selection_string: str,
    name: str,
) -> dict:
    """Loads MD trajectory paths for a given boxes path"""
    path_dict = {}
    for i in range(nboxes):
        path_dict[f"{name}_{i}"] = {
            "pdb": os.path.join(
                os.path.join(boxes_path, f"sim{i}"), "equilibrated.gro"
            ),
            "xtc": os.path.join(os.path.join(boxes_path, f"sim{i}"), "traj_comp.xtc"),
            "tpr": os.path.join(os.path.join(boxes_path, f"sim{i}"), "topol.tpr"),
            "bp_selection_string": bp_selection_string,
        }
    return path_dict


def load_abfe_paths_for_compound(
    # TODO: Need to change this not have old and new vanilla paths for van1
    config: dict,
    cmp: str,
    bp_selection_string: str,
    van_list: list[str],
    leg_window_list: list[str],
    apo=True,
    apo_list: list[int] = [1, 2, 3],
    vanilla_path_template_old=False,
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
                "tpr": os.path.join(
                    config["abfe_window_path_template"].format(
                        CMP_NAME=cmp,
                        REP_NO=van_rep_no,
                        STAGE="prod",
                        ABFE_REP_NO=1,
                        LEG_WINDOW=leg_window,
                    ),
                    "topol.tpr",
                ),
                "bp_selection_string": bp_selection_string,
            }
            path_dict[f"{cmp}_nvt"] = {
                "pdb": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1 and vanilla_path_template_old
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "nvt.gro",
                ),
                "xtc": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1 and vanilla_path_template_old
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "nvt.xtc",
                ),
                "tpr": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1 and vanilla_path_template_old
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "nvt.tpr",
                ),
                "bp_selection_string": bp_selection_string,
            }
            path_dict[f"{cmp}_van_{van_rep_no}"] = {
                "pdb": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1 and vanilla_path_template_old
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "npt.gro",
                ),
                "xtc": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1 and vanilla_path_template_old
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "prod.xtc",
                ),
                "tpr": os.path.join(
                    config[
                        "vanilla_path_template_old"
                        if van_rep_no == 1 and vanilla_path_template_old
                        else "vanilla_path_template"
                    ].format(CMP_NAME=cmp, REP_NO=van_rep_no),
                    "prod.tpr",
                ),
                "bp_selection_string": bp_selection_string,
            }
    if apo:
        for apo_rep_no in apo_list:
            path_dict[f"apo_{apo_rep_no}"] = {
                "pdb": os.path.join(
                    config["apo_path_template"].format(REP_NO=apo_rep_no), "npt.gro"
                ),
                "xtc": os.path.join(
                    config["apo_path_template"].format(REP_NO=apo_rep_no), "prod.xtc"
                ),
                "tpr": os.path.join(
                    config["apo_path_template"].format(REP_NO=apo_rep_no), "prod.tpr"
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
            "tpr": os.path.join(
                config["apo_path_template"].format(REP_NO=apo_rep_no), "prod.tpr"
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
