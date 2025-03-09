import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF,  DCD,  GRO,  XTC
import warnings
from matplotlib import pyplot as plt
from MDAnalysis.lib import distances
# first  import nglview
import nglview as nv
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.cluster import KMeans
import hdbscan
from sklearn import preprocessing
import umap
from scipy.spatial.distance import jensenshannon
from trajDimRed_v2 import *

pocket_residues = '12  54  57  58  59  60  61  62  64  65  66  71  77  78  81  82  83  84  85  86  89  90  135  138  141  142  161  162  163  174  175  178  182  232  235  236  238  239  242  254  261  264  265  266  268  269'

cmp1_prod = annotate_BP(gro='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/OX2_43289/vanilla/npt.gro',
                            xtc='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/OX2_43289/vanilla/prod.xtc',
                            selection_string=f"protein and resid {pocket_residues}")

cmp1_closest = annotate_BP(gro='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/OX2_43289/vanilla/em.gro',
                            selection_string=f"protein and resid {pocket_residues}")


# Add long apo+cmp1 sims to universe dict
long_apo_r1 = annotate_BP(gro='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/apo_OX2_r1/npt.gro',
                            xtc='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/apo_OX2_r1/prod.xtc',
                            selection_string=f"protein and resid {pocket_residues}")
long_apo_r2 = annotate_BP(gro='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/apo_OX2_r2/npt.gro',
                            xtc='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/apo_OX2_r2/prod.xtc',
                            selection_string=f"protein and resid {pocket_residues}")
long_apo_r3 = annotate_BP(gro='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/apo_OX2_r3/npt.gro',
                            xtc='/biggin/b211/reub0138/Projects/orexin/deflorian_set_1_j13_v1/apo_OX2_r3/prod.xtc',
                            selection_string=f"protein and resid {pocket_residues}")

# Make universe dict out of equillibriation sims
universe_dict = {"cmp1_prod":cmp1_prod,'cmp1_closest':cmp1_closest,
                 'long_apo_r1':long_apo_r1,'long_apo_r2':long_apo_r2,'long_apo_r3':long_apo_r3}


# Create an instance of the TrajectoryAnalyzer class and analyze
analyzer = trajDimRed(universe_dict,saved=True)
analyzer.do_dimred(method='pca',n_components=6)
#analyzer.do_dimred(method='umap',n_components=2,n_neighbors=100,min_dist=0.1)
analyzer.plot_dimred_sims(method='pca',  targets=None,  highlights=['cmp1_closest'],  save_path='pca_sims_v1.png')
# #analyzer.plot_dimred_sims(method='umap'  targets=None  highlights=None  save_path='umap_sims_v0.png')
analyzer.plot_dimred_time('long_apo_r1',  method='pca',  save_path='pca_time_v1_apor1.png')
analyzer.plot_dimred_time('long_apo_r2',  method='pca',  save_path='pca_time_v1_apor2.png')
analyzer.plot_dimred_time('long_apo_r3',  method='pca',  save_path='pca_time_v1_apor3.png')
# #analyzer.plot_dimred_time('long_cmp1_r1'  method='umap'  save_path='umap_time_v0.png')
analyzer.elbow_plot(method='pca',save_path='elbow_plot_v1.png')
analyzer.cluster(6,  method='pca')
# analyzer.cluster(4,method='umap')
analyzer.plot_dimred_cluster(clusters=None  ,method='pca'  ,palette = "husl"  ,alpha=1  ,s=5  ,save_path='pca_cluster.png')

analyzer.compute_jensen_shanon('cluster',1,2,num_bins=50  ,method='pca')
analyzer.plot_top_histograms('cluster',1,2,top_n=20,residoffset=0,method = 'pca'  ,save_path='js_cluster_pca.png')
# #analyzer.compute_jensen_shanon('cluster',1,2,num_bins=50  ,method='umap')
# #analyzer.plot_top_histograms('cluster',1,2,top_n=20,residoffset=6,method = 'umap'  ,save_path='js_cluster_umap.png')
u_closest = analyzer.read_sliced_trajectory('simulation'  ,'cmp1_closest'  ,method='pca')

# #analyzer.plot_rmsf_subset('simulation','long_cmp1_r1','long_apo_r1',method='pca',residoffset=0,ref = u_closest)
# #analyzer.plot_top_histograms_w_restraints('simulation','long_cmp1_r1','long_apo_r1',save_path='js_sims_cmp1_prod_long_cmp1_r1.png')
# # analyzer.plot_top_histograms_w_restraints('simulation','cmp1_prod','long_apo_r1',ref1=u_closest,save_path='js_closest_sims_cmp1_prod_long_cmp1_r1.png',a=1)
# # analyzer.plot_top_histograms_w_restraints('simulation','cmp1_prod','long_apo_r2',ref1=u_closest,save_path='js_closest_sims_cmp1_prod_long_cmp1_r1.png',a=1)

# # analyzer.generate_gmx_restraints_file(rmsf_list=[1 for i in range(len(pocket_residues.split('  ')))],ref=u_closest,save_path='restraints.dat')

# # Create an instance of the TrajectoryAnalyzer class and analyze the trajectories
# #analyzer = trajDimRed(universe_dict)
# #analyzer.check_atom_residue_consistency()
# #analyzer.analyze()
# # nneighbours -> Low: Local
# # nneighbours -> High: Global
# # Min dist -> Low; dense packing local structure
# # Min dist -> High; loose packing global s tructure
# #analyzer.do_dimred(method='pca',n_components=6)
# #analyzer.cluster(7,'pca')
# #analyzer.plot_dimred_cluster(method='pca',alpha=0.5,s=8,palette='husl',save_path='pca_cluster_v0.png')
# #analyzer.compute_jensen_shanon('sim','cmp1_prod','long_apo_r1',method = 'pca')
# #analyzer.compute_jensen_shanon('sim','cmp1_prod','long_apo_r2',method = 'pca')
# #analyzer.compute_jensen_shanon('sim','cmp1_prod','long_cmp1_r1',method = 'pca')
# #analyzer.plot_top_histograms('sim','cmp1_prod','long_apo_r1',8,49,method='umap',save_path='js_sims_cmp1_prod_long_apo_r1.png')
# #analyzer.plot_top_histograms('sim','cmp1_prod','long_apo_r2',8,49,method='umap',save_path='js_sims_cmp1_prod_long_apo_r2.png')
# #analyzer.plot_top_histograms('sim','cmp1_prod','long_cmp1_r1',8,49,method='umap',save_path='js_sims_cmp1_prod_long_cmp1_r1.png')
# #analyzer.cluster(8,'umap')