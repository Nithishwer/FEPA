This tutorial is designed to get the user familiar with fepa and its functionality. We will be analysing MD data from a set of ABFE simulations for the OX2R receptor to a single compound out of the 27 ligands from Deflorian et al set 1. We will featurize the binding pockets of the receptor in the holo, apo and abfe simulations, identify apo state sampling issues in the ABFE simulations and setup REUS simulations to correct for the systematic overestimation from apo state sampling inaccuracies. This tutorial uses MDAnalysis, MDtraj and the PENSA package for analysis, GROMACS for simulation and Plumed for enhanced sampling. It is assumed that the user is familiar with setting up and analyzing MD simulations run with GROMACS and Plumed. 

# Loading the config file

ABFE simulations typically generate multiple MD trajectories over different lamba windows. In our case, we have 44 lambda windows (11 Coulomb, 12 Restrains and 22 Van-der-Walls) and we also have the holo and the apo simulations that we have to analyse. To make things easier when parsing paths to the topology and coordinate files of these trajectories, we use a config file. The config file is json-formatted and contains all the information necessary to read and analyze the simulations. Here is a sample config file to be used with this tutorial:

```json
{
    "base_path": "deflorian_set_1_j13_v1",
    "abfe_window_path_template": "deflorian_set_1_j13_v1/OX2_{CMP_NAME}/abfe_van{REP_NO}_hrex_r{ABFE_REP_NO}/complex/{LEG_WINDOW}/{STAGE}",
    "vanilla_path_template": "deflorian_set_1_j13_v1/OX2_{CMP_NAME}/vanilla_rep_{REP_NO}",
    "apo_path_template": "deflorian_set_1_j13_v1/apo_OX2_r{REP_NO}",
    "compounds": [
        "42922",
    ],
    "pocket_residues_string": "12  54  57  58  59  60  61  62  63  64  65  70  71  78  81  82  83  85  86  89 138 142 160 161 162 163 175 178 179 182 183 232 235 236 239 240 242 243 261 265 268 269"
}
```

The pocket_residues_string variable is a string that stores the residue ids of all the proteins residue that have any atom within 6 A of the ligand. The JSON file is then loaded into a dictionary with the function `load_config` from `fepa.utils.file_utils`

# Loading MD trajectories with EnsembleHandler

Now that we have the template paths to all the simulations: Apo EQ, Holo EQ and ABFE, we can use the function `load_abfe_paths_for_compound` from fepa.utils.path_utils to generate a `path_dict` dictionary for a single compound as follows:

```python
cmp = config['compounds'][0]
path_dict = load_abfe_paths_for_compound(
            config,
            cmp,
            van_list=[1, 2, 3],
            leg_window_list=[f"coul.{i:02d}" for i in range(0, 11)] + [f'vdw.{i:02d}' for i in range(0, 21)] + [f'rest.{i:02d}' for i in range(0, 12)],
            bp_selection_string="name CA and resid " + config["pocket_residues_string"],
            apo=True,
        )
```

The `load_abfe_paths_for_compound` function has various arguments that allow the user control over what simulation paths are loaded. For more information, please refer to the API. 

Now that we have the path_dict, it is time for us to load the trajectories themselves. We will be using the EnsembleHandler class from fepa.core.ensemble_handler to do this. EnsembleHandler is a neat way of storing and manipulating the trajectories from multiple ensembles with some built in functions for sanity checks. Internally EnsembleHandler stores the trajectories as dictionary of MDA universes.  

NOTE: Need to remove ensemblehandler and just use universe dict

```python
# Load trajectories
ensemble_handler = EnsembleHandler(path_dict)
# Make universes
logging.info("Making universes for compound %s...", cmp)
# Check for BP residue consistency across all the trajectories
logging.info("Checking residue consistency for compound %s...", cmp)
```

# Featurizing

We will now be featurizing these trajectories by their pairwise C-alpha distances in the binding pocket using the function `SelfDistanceFeaturizer` from fepa.core.featurizers. `SelfDistanceFeaturizer` computes and stored all possible pairs of distances between the C-alpha atoms of the binding pocket residues. The class also has `save_features` and `load_features` functions that help save the features as a csv file to make sure the time consuming featurization step need not be repeated every run.

```
# Make a folder for the analysis output
cmp_run_dir = f'analysis/{cmp}/'
if !(os.path.exists(cmp_run_dir)):
    os.mkdir(cmp_run_dir)

# Featurize and save features
featurizer = SelfDistanceFeaturizer(ensemble_handler)
featurizer.featurize()
featurizer.save_features(input_dir=cmp_existing_run_dir)
```

# Visualizing the ensembles

Saving the features in a csv format gives us the flexibility to analyse it as required. For the purpose of this tutorial, we will be looking at how our features capture the difference between the apo, the holo and the abfe ensembles. To do this we reduce the dimensions of the features data using the `PCADimReducer` class from `fepa.core.dim_reducers`. FEPA also supports other dimensionality reduction techniqeus like UMAP and tSNE. In fact UMAP is better able to resolve the differences in binding pocket configurations between different ensembles. We will be doing PCA here as it also doubles up as a nice CV to bias when performing umbrella sampling later.

```
# Dimensionality Reduction
logging.info("Performing dimensionality reduction for compound %s...", cmp)
dimreducer = PCADimReducer(featurizer.get_feature_df(), n_components=8)
dimreducer.reduce_dimensions()
dimreducer.calculate_projections()
dimreducer.save_projection_df(
    save_path=os.path.join(cmp_output_dir, "pca_projection_df.csv")
)
```

First we plot the eigen values of all the PCs to understand what percentage of variance is captured by the first few PCs:

```
logging.info("Plotting PCA eigenvalues for compound %s...", cmp)
plot_eigenvalues(
    pca_object=dimreducer.get_pca(),
    n_components=8,
    save_path=os.path.join(cmp_output_dir, "eigenvalues.png"),
)
```

We can then visualize the dimensionality reduced data using the DimRedVisualizer class. This class allows us to visualize all aspects of the data without having to rewrite functions all the time. In the code snippet given below, we plot the first two PCs colored by simulation and time. The class also allows us to send list of data points that must be highlighted.

```
# Visualization
projection_df = dimreducer.get_pca_projection_df()
# remove rows with ensemble containing nvt
projection_df = projection_df[~projection_df["ensemble"].str.contains("apo")]
logging.info("Visualizing compound %s...", cmp)
dimred_visualizer = DimRedVisualizer(
    projection_df=projection_df, data_name="PCA"
)
dimred_visualizer.plot_dimred_sims(
    save_path=os.path.join(cmp_output_dir, "pca_components_ensemble_noapo.png"),
    highlights=[f"{cmp}_nvt"],
)
dimred_visualizer.plot_dimred_time(
    save_path=os.path.join(cmp_output_dir, "pca_components_time_noapo.png")
)
```

