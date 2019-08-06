from __future__ import division
import numpy as np


kin_file = 'data/DG1_kin.csv'
input_file = 'data/full_sample_6_kin_vars.csv'
final_sample_file = 'data/final_sample.csv'
clusters_file = 'clusters.py'

dispersion_distrib_file_mask = ['data/dispersion_distribs/dispersion_distrib_','.hdf5']
dispersion_distrib_plot_file_mask = ['data/dispersion_distribs/plots/dispersion_distrib_','.png']
cluster_dispersion_plot_file_mask = ['plots/cluster_dispersions/cluster_dispersion_','.png']
dispersion_cluster_size = [2,6]

colors = ['red', 'blue', 'green', 'black', 'purple', 'yellow', 'teal', 'red', 'blue', 'green', 'black', 'purple', 'yellow', 'teal', 'red', 'blue', 'green', 'black', 'purple', 'yellow', 'teal', 'red', 'blue', 'green', 'black', 'purple', 'yellow', 'teal', 'red', 'blue', 'green', 'black', 'purple', 'yellow', 'teal', 'red', 'blue', 'green', 'black', 'purple', 'yellow', 'teal']
markers = ['+', '1', '2', 'x', '*', '.', '^', '1', '2', 'x', '*', '.', '^', '+', '2', 'x', '*', '.', '^', '+', '1', 'x', '+', '1', '2', 'x', '*', '.', '^', '1', '2', 'x', '*', '.', '^', '+', '2', 'x', '*', '.', '^', '+']
