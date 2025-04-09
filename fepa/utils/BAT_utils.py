import MDAnalysis as mda
from fepa.utils.coord_trans import BAT


def read_BAT(
    pdb,
    xtc,
    sel,
    first_frame=0,
    last_frame=None,
    step=1,
    naming=None,
):
    """
    Load Bond Angles and Torsions coordinates for a given selection

    Parameters
    ----------
    pdb : str
        File name for the reference file (PDB or GRO format).
    xtc : str
        File name for the trajectory (xtc format).
    sel : str
        Selection string for the atoms to be used.
    first_frame : int, default=0
        First frame to return of the features. Zero-based.
    last_frame : int, default=None
        Last frame to return of the features. Zero-based.
    step : int, default=1
        Subsampling step width when reading the frames.
    naming : str, default='plain'
        Naming scheme for each atom in the feature names.
        plain: neither chain nor segment ID included
        chainid: include chain ID (only works if chains are defined)
        segid: include segment ID (only works if segments are defined)
        segindex: include segment index (only works if segments are defined)

    Returns
    -------
    feature_names : list of str
        Generic names of all torsions
    features_data : numpy array
        Data for all torsions [Å]
    """

    # Read the dihedral angles
    u = mda.Universe(pdb, xtc)

    # Select residues
    selection = u.select_atoms(sel)

    # Calculate BAT coordinates for a trajectory
    R = BAT(selection)
    R.run()

    bat_array = R.results["bat"]
    bat_names = ["BAT_" + name for name in R.results["names"]]

    return bat_names, bat_array[first_frame:last_frame:step]
