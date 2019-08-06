from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import h5py
from sklearn.neighbors import KernelDensity
import params
import clusters


sigma_1_frac = 0.68
sigma_2_frac = 0.95
sigma_3_frac = 0.997

cluster_IDs = ['c'+str(i) for i in range(50)]

bandwidth_refinement = 5


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def plot_dispersion_distrib(cluster_ID, cluster, Vars, var_names, var_labels):
    if len(Vars)>1:
        for i in range(len(Vars)):
            Vars[i] = (Vars[i] - np.mean(Vars[i])) / np.std(Vars[i])
    Vars = np.transpose(Vars)
    Vars = Vars[cluster]
    
    filename = params.dispersion_distrib_file_mask[0] + var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += params.dispersion_distrib_file_mask[1]

    i = len(Vars)
    f = h5py.File(filename, 'r')
    a = f["/dispersion"][i-params.dispersion_cluster_size[0]]
    N_samples = len(a)
    
    x = np.linspace(min(a), max(a), N_samples)
    kde = KernelDensity(kernel='gaussian', bandwidth=(max(a)-min(a))/100/bandwidth_refinement).fit(a[:, None])
    a_kde = kde.score_samples(x[:, None])
    a_kde = np.exp(a_kde)

    plt.clf()
    plt.title("Dispersion distribution, " + str(i+params.dispersion_cluster_size[0]) + " stars", size=24)

    name = var_labels[0]
    if len(var_labels)>1:
        for v in var_labels[1:]:
            name += (', ' + v)
    name += ' standard deviation'
    plt.xlabel(name, size=24)

    plt.ylabel('Probability', size=24)
    plt.tick_params(labelsize=18)
    plt.fill_between(x, 0, a_kde, color='grey')

    sigma_line_height = N_samples/sum(a)
    a = sorted(a)
    plt.plot ([a[int((1-sigma_1_frac)*N_samples-1)], a[int((1-sigma_1_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='black', linewidth=2)
    plt.plot ([a[int((1-sigma_2_frac)*N_samples-1)], a[int((1-sigma_2_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='blue', linewidth=2)
    plt.plot ([a[int((1-sigma_3_frac)*N_samples-1)], a[int((1-sigma_3_frac)*N_samples-1)]], [0, sigma_line_height], '--', color='red', linewidth=2)

    plt.scatter (np.std(Vars), sigma_line_height/2, s=200, color=params.colors[cluster_ID], marker=params.markers[cluster_ID])

    plt.gcf().set_size_inches(25.6, 14.4)
        
    filename = params.cluster_dispersion_plot_file_mask[0] + var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += ('_'+cluster_IDs[cluster_ID])
    filename += params.cluster_dispersion_plot_file_mask[1]
    plt.gcf().savefig(filename, dpi=100)
    plt.close()


def plot_all_cluster_dispersions(Clusters, Vars, var_names, var_labels, markersize=200):
    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("Dispersion confidence levels for d"+var_labels[0], size=24)
    plt.ylabel("Clusters", size=24)
    plt.xlim(0, 1)
    plt.ylim(0, 1440) 
    plt.tick_params(labelsize=18)
    plt.gca().set_yticklabels([])

    N, Nv = len(Clusters), len(Vars)
    for i, v in enumerate(Vars):
        for j, c in enumerate(Clusters):
            # Calculate the position of the box on the plot.
            delta_x = 2540/Nv - 20
            delta_y = 1420/N - 20
            x1 = (i+1)*20 + i*delta_x
            x2 = (i+1)*20 + (i+1)*delta_x
            y1 = (j+1)*20 + j*delta_y
            y2 = (j+1)*20 + (j+1)*delta_y
            # Retrieve the dispersion array.
            filename = params.dispersion_distrib_file_mask[0] + var_names[i] + params.dispersion_distrib_file_mask[1]        
            f = h5py.File(filename, 'r')
            Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
            N_samples = len(Dispersion_array)
            # Calculate the variable dispersion in the cluster.
            var = np.transpose(v)
            var = var[c]
            var_std = np.std(var)
            # Find the variable location on the plot on the scale 0 to 1.
            point_x = x1 + (1/2 + (len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples)*delta_x
            # Plot everything.
            plt.plot([x1/2560,x2/2560], [y1,y1], color='black', linewidth=1)
            plt.plot([x1/2560,x2/2560], [y2,y2], color='black', linewidth=1)
            plt.plot([x1/2560,x1/2560], [y1,y2], color='black', linewidth=1) 
            plt.plot([x2/2560,x2/2560], [y1,y2], color='black', linewidth=1)
            plt.plot([(x2 - delta_x*sigma_1_frac)/2560, (x2 - delta_x*sigma_1_frac)/2560], [y1,y2], '--', color='black', linewidth=1)
            plt.plot([(x2 - delta_x*sigma_2_frac)/2560, (x2 - delta_x*sigma_2_frac)/2560], [y1,y2], '--', color='blue', linewidth=1)
            plt.plot([(x2 - delta_x*sigma_3_frac)/2560, (x2 - delta_x*sigma_3_frac)/2560], [y1,y2], '--', color='red', linewidth=1)
            plt.scatter(point_x/2560, (y1+y2)/2, color=params.colors[j], s=markersize, marker=params.markers[j])
    # Save the plot.
    filename = 'plots/all_cluster_dispersions_'+var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += '.png'
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig(filename, dpi=100)
    plt.close()


def plot_all_cluster_dispersions_2D(Clusters, Vars, var_names, var_labels, markersize=200, ylimits = [1,0]):
    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("<"+var_labels[0]+">", size=24)
    plt.ylabel("Dispersion confidence levels for d"+var_labels[1], size=24)
    C = []
    for c in Clusters:
        C.append(np.mean(list(Vars[0][c].flatten())))
    plt.xlim(min(C)-abs(max(C)-min(C))/20, max(C)+abs(max(C)-min(C))/20)
    plt.ylim(ylimits[0],ylimits[1])
    plt.tick_params(labelsize=18)
    plt.plot([min(Vars[0].flatten())-abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2, max(Vars[0].flatten())+abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2], [1-sigma_1_frac, 1-sigma_1_frac], '--', color='black', linewidth=1)
    plt.plot([min(Vars[0].flatten())-abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2, max(Vars[0].flatten())+abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2], [1-sigma_2_frac, 1-sigma_2_frac], '--', color='blue', linewidth=1)
    plt.plot([min(Vars[0].flatten())-abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2, max(Vars[0].flatten())+abs(max(Vars[0].flatten())-min(Vars[0].flatten()))/2], [1-sigma_3_frac, 1-sigma_3_frac], '--', color='red', linewidth=1)

    for j, c in enumerate(Clusters):
            # Retrieve the dispersion array.
            filename = params.dispersion_distrib_file_mask[0] + var_names[1] + params.dispersion_distrib_file_mask[1]        
            f = h5py.File(filename, 'r')
            Dispersion_array = f["/dispersion"][len(c)-params.dispersion_cluster_size[0]]
            N_samples = len(Dispersion_array)
            # Calculate the variable dispersion in the cluster.
            var = np.transpose(Vars[1])
            var = var[c]
            var_std = np.std(var)
            # Find the variable location on the plot on the scale 0 to 1.
            point_x = np.mean(Vars[0][c])
            point_y = 1/2+(len(Dispersion_array[Dispersion_array<var_std])+1-len(Dispersion_array[Dispersion_array>var_std]))/2/N_samples 
            plt.scatter(point_x, point_y, color=params.colors[j], s=markersize, marker=params.markers[j])

    # Save the plot
    filename = 'plots/all_cluster_dispersions_2D_'+var_names[0]
    if len(var_names)>1:
        for v in var_names[1:]:
            filename += ('_' + v)
    filename += '.png'
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig(filename, dpi=100)
    plt.close()

if __name__ == '__main__':
    file_input = params.final_sample_file
    FeH = get_column(1, float, file_input)
    SrBa = get_column(3, float, file_input)
    BaEu = get_column(4, float, file_input)
    EuFe = get_column(5, float, file_input)
    A_C = get_column(6, float, file_input) 
    CFe = get_column(2, float, file_input)
    vRg = get_column(12, float, file_input)
    vTg = get_column(13, float, file_input)
    r_apo = get_column(14, float, file_input)
    Clusters = clusters.final_clusters

    #for i in range(0, len(Clusters)):
    #    plot_dispersion_distrib(i, Clusters[i], [FeH], ['FeH'], ['[Fe/H]'])
    #    plot_dispersion_distrib(i, Clusters[i], [A_C], ['AC'], ['A(C)'])
    #    plot_dispersion_distrib(i, Clusters[i], [FeH, A_C], ['FeH', 'AC'], ['[Fe/H]', 'A(C)'])
    #    plot_dispersion_distrib(i, Clusters[i], [CFe], ['CFe'], ['[C/Fe]'])
    #    plot_dispersion_distrib(i, Clusters[i], [vRg], ['vRg'], ['vRg'])
    #    plot_dispersion_distrib(i, Clusters[i], [vTg], ['vTg'], ['vTg'])
    #    plot_dispersion_distrib(i, Clusters[i], [r_apo], ['r_apo'], ['r_apo'])
    plot_all_cluster_dispersions(Clusters, [FeH], ['FeH'], ['[Fe/H]'], markersize=600)
    plot_all_cluster_dispersions(Clusters, [A_C], ['AC'], ['A(C)'], markersize=600)
    plot_all_cluster_dispersions(Clusters, [r_apo], ['r_apo'], ['r_apo'], markersize=600)
    plot_all_cluster_dispersions(Clusters, [CFe], ['CFe'], ['[C/Fe]'], markersize=600)
    plot_all_cluster_dispersions(Clusters, [SrBa], ['SrBa'], ['[Sr/Ba]'], markersize=600)
    plot_all_cluster_dispersions(Clusters, [BaEu], ['BaEu'], ['[Ba/Eu]'], markersize=600)
    plot_all_cluster_dispersions(Clusters, [EuFe], ['EuFe'], ['[Eu/Fe]'], markersize=600)
    plot_all_cluster_dispersions_2D(Clusters, [EuFe,FeH], ['EuFe','FeH'], ['[Eu/Fe]','[Fe/H]'], markersize=600, ylimits=[1,0])
    plot_all_cluster_dispersions_2D(Clusters, [FeH,EuFe], ['FeH','EuFe'], ['[Fe/H]','[Eu/Fe]'], markersize=600, ylimits=[0.06,0])
    plot_all_cluster_dispersions_2D(Clusters, [EuFe,CFe], ['EuFe','CFe'], ['[Eu/Fe]','[C/Fe]'], markersize=600, ylimits=[0.10,0])
