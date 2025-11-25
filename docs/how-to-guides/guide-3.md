---
icon: lucide/compass
title: Ligand conformations
---

This workflow shows how FEPA could be used to evaluate binding-pose similarity across ensembles. To do this, we will load ABFE trajectories, cluster binding poses using MDAnalysis Encore, and export both cluster assignments and representative centroid structures for visualization.

### 1. Loading ABFE Trajectories

We begin by loading the trajectories for a given compound and initializing the `EnsembleHandler` object:

```python
import logging, os
from fepa.utils.file_utils import load_config
from fepa.utils.path_utils import load_abfe_paths_for_compound
from fepa.core.ensemble_handler import EnsembleHandler

# Load configuration
config_path = os.path.join("../../config/config.json")
config = load_config(config_path)
analysis_output_dir = "./wdir"
os.makedirs(analysis_output_dir, exist_ok=True)

van_list = [1, 2, 3]
leg_window_list = (
    [f"coul.{i:02}" for i in range(0, 11, 2)]
    + [f"vdw.{i:02}" for i in range(0, 12, 2)]
    + [f"rest.{i:02}" for i in range(0, 11, 2)]
)

path_dict = load_abfe_paths_for_compound(
    config,
    cmp=cmp,
    bp_selection_string="name CA and resid " + config["pocket_residues_string"],
    van_list=van_list,
    leg_window_list=leg_window_list,
    apo=False,
)

# Load trajectories
ensemble_handler = EnsembleHandler(path_dict)
ensemble_handler.make_universes()

check_bp_residue_consistency(ensemble_handler.get_universe_dict())
```

Here we collect all holo ensembles, ensure pocket residues are consistent across trajectories, and prepare the universes for analysis.

### 2. Clustering Ligand Binding Poses

We then perform binding-pose clustering using DBSCAN from MDAnalysis Encore. Each cluster represents a distinct ligand binding mode shared across the ensemble set.

```python
sel = "resname unk"
universe_dict = ensemble_handler.get_universe_dict()
# Define clustering method
dbscan_method = encore.DBSCAN(
    eps=0.5, min_samples=5, algorithm="auto", leaf_size=30
)
# Define the ensemble list
ensemble_list = list(universe_dict.values())
# Cluster the binding poses
cluster_collection = encore.cluster(
    ensembles=ensemble_list,
    select=sel,
    superimposition_subset="name CA",
    method=dbscan_method,
)
```
This aligns the complexes on Cα atoms and clusters ligand heavy-atom coordinates to identify recurring conformations.


### 3. Saving Framewise Cluster Assignments

The cluster identity for each frame is stored in a DataFrame and written to disk:

```python
cluster_df = pd.DataFrame({
    "timestep": timstep_series,
    "ensemble": ensemble_series,
    "cluster": cluster_series,
})
cluster_df.to_csv(
    os.path.join(cmp_output_dir, f"{cmp}_conformation_cluster_df.csv"),
    index=False,
)
```

### 4. Exporting Cluster Centroid Structures

For visualization in PyMOL or ChimeraX, we extract each cluster’s centroid frame as a PDB file:

```python
ensemble_handler.dump_frames(
    ensemble=centroid_ensemble,
    timestep=centroid_timestep,
    save_path=os.path.join(
        cmp_output_dir,
        f"{cmp}_conformation_cluster_{cluster_id}.pdb",
    ),
)
```

These structures serve as representative snapshots of each binding mode.

### 5. Visualizing Cluster Populations

We can summarize cluster occupancy across ensembles using a simple stacked bar plot:

```python
# Plot a stacked barplot of clusters in each ensemble
cluster_counts = (
    cluster_df.groupby(["ensemble", "cluster"]).size().unstack(fill_value=0)
)
cluster_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    colormap="Set2",
)
plt.title(f"Cluster distribution in each ensemble for {cmp}")
plt.xlabel("Ensemble")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(
    os.path.join(
    analysis_output_dir,
    f"{cmp}_conformation_cluster_distribution.png",
    )
)
plt.close()
```

This visualization highlights whether certain λ-windows prefer specific ligand poses, helping diagnose convergence and structural consistency across ABFE simulations.

![Figure-1](image-3.png)
**Figure 1.** Stacked bar plot of conformations for ligand 46853 (OX2R, Deflorian et al set 1) across ABFE windows for three ABFE runs. The ligand adopts one pose in run 1 and a different pose in runs 2 and 3.

![Figure-2](image-4.png)
**Figure 2.** Visualization of the PDBs shows that while the binding poses in clusters 0 and 1 are crystal-like and similar to each other, they differ in the orientation of the triazole ring. Cluster 2, however, is very different. Fortunately, cluster 2 is rare in the simulations, so the calculations predominantly reflect the crystal-like pose.
