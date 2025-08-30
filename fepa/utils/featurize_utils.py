from typing import Iterable, List, Optional, Tuple
import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import self_distance_array

def compute_self_distances_with_transforms(
    tpr_path: str,
    xtc_path: str,
    selection: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    transformations: Optional[Iterable] = None,
    pbc: bool = True,
    feature_prefix: str = "DIST",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a Universe, optionally apply MDAnalysis transformations, and compute all unique i<j
    pairwise distances within the atom selection for each frame.

    Returns
    -------
    names : (n_pairs,) object array
        "DIST: RES RESID ATOM - RES RESID ATOM" per i<j pair (matches downstream parsing).
    data : (n_frames, n_pairs) float32 array
        Distances per frame (Å), minimum-image if box available and pbc=True.
    """
    u = mda.Universe(tpr_path, xtc_path)

    if transformations:
        # avoid stacking the same transforms repeatedly
        if not hasattr(u.trajectory, "_fepa_transforms_added"):
            u.trajectory.add_transformations(*transformations)
            setattr(u.trajectory, "_fepa_transforms_added", True)

    ag: AtomGroup = u.select_atoms(selection)
    n = ag.n_atoms
    if n < 2:
        raise ValueError(f"Selection '{selection}' contains fewer than 2 atoms.")

    # Build pair names consistent with your utils parsing
    atoms = ag.atoms
    names: List[str] = []
    for i in range(n - 1):
        ai = atoms[i]
        for j in range(i + 1, n):
            aj = atoms[j]
            names.append(
                f"{feature_prefix}: {ai.resname} {ai.resid} {ai.name} - {aj.resname} {aj.resid} {aj.name}"
            )
    names = np.array(names, dtype=object)

    use_box = pbc and (u.dimensions is not None) and (u.dimensions.size >= 3)
    rows: List[np.ndarray] = []
    for _ in u.trajectory[slice(start, stop, step)]:
        coords = ag.positions.astype(np.float32, copy=False)
        box = u.dimensions if use_box else None
        d = self_distance_array(coords, box=box).astype(np.float32, copy=False)
        rows.append(d)

    data = np.vstack(rows) if rows else np.empty((0, len(names)), dtype=np.float32)
    return names, data
