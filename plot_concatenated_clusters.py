from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import itertools
import os
import astropy
import astropy.units as u
import astropy.coordinates as coord
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn import datasets
from sklearn.manifold import TSNE
import params
import clusters


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)
  

def plot_data(name, var_name_1, var_name_2, var1, var2):
    plt.clf()
    plt.title("r-process sample", size=24)
    plt.xlabel(var_name_1, size=24)
    plt.ylabel(var_name_2, size=24)
    plt.tick_params(labelsize=18)
    plt.scatter(var1, var2, c='black', s=50)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/"+name, dpi=100)
    plt.close()


def plot_concatenated_clusters (name, Clusters, colors, markers, var_name_1, var_name_2, var1, var2, show_all, Xlim=[0,0], Ylim=[0,0], markersize=(10,200)):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(var1)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel(var_name_1, size=24)
    plt.ylabel(var_name_2, size=24)
    if Xlim[0] < Xlim[1]:
        plt.xlim(Xlim[0], Xlim[1])
    if Ylim[0] < Ylim[1]:
        plt.ylim(Ylim[0], Ylim[1])
    plt.tick_params(labelsize=18)
    for c, color, marker in zip(Clusters, colors, markers):
        plt.scatter(var1[c], var2[c], c=color, marker=marker, s=markersize[1])
    if show_all==True:
        plt.scatter(var1[non_indeces], var2[non_indeces], c='grey', marker='o', s=markersize[0])
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/"+name, dpi=100)
    plt.close()


def plot_mean_metallicities_RV (Clusters, colors, FeH, RV):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(FeH)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("<[Fe/H]>", size=24)
    plt.ylabel("dRV (km/s)", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        FeH_avg = np.mean(FeH[c])
        RV_std = np.std(RV[c])
        plt.scatter (FeH_avg, RV_std, c=colors[i], marker=params.markers[i], s=400)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/Metallicities_RV.png", dpi=100)
    plt.close()    


def plot_orbital_velocities (Clusters, vRg, vTg):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(vRg)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("vRg", size=24)
    plt.ylabel("vTg", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(vRg[c], vTg[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.scatter(vRg[non_indeces], vTg[non_indeces], c='grey', marker='o', s=10)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/Orbital_velocities_clusters.png", dpi=100)
    plt.close() 


def plot_vTg_rapo (Clusters, vTg, r_apo):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(r_apo)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("r_apo", size=24)
    plt.ylabel("vTg", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(r_apo[c], vTg[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/vTg_rapo_clusters.png", dpi=100)
    plt.close() 


def plot_coordinates (Clusters, l, b):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(l)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("l", size=24)
    plt.ylabel("b", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(l[c], b[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.scatter(l[non_indeces], b[non_indeces], c='grey', marker='o', s=10)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/coordinates_clusters.png", dpi=100)
    plt.close() 


def plot_distances_RV (Clusters, dist, RV):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    non_indeces = [i for i in range(len(RV)) if not (i in indeces)]

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel("dist", size=24)
    plt.ylabel("RV", size=24)
    plt.tick_params(labelsize=18)
    for i, c in enumerate(Clusters):
        plt.scatter(dist[c], RV[c], c=params.colors[i], marker=params.markers[i], s=200)
    plt.scatter(dist[non_indeces], RV[non_indeces], c='grey', marker='o', s=10)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/dist_RV_clusters.png", dpi=100)
    plt.close() 


def plot_t_SNE (Clusters, Params, name, markersize = [10,250]):
    indeces = [c[i] for c in Clusters for i in range(len(c))]
    for i in range(len(Params)):
        Params[i] = (Params[i] - np.mean(Params[i]))/np.std(Params[i])
    non_indeces = [i for i in range(len(Params[0])) if not (i in indeces)]

    Params = np.transpose(Params)
    TSNE_coord_clusters = np.transpose(TSNE(n_components=2).fit_transform(Params))

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel ("t-SNE plot", size=24)
    plt.ylabel("", size=24)
    plt.tick_params(labelsize=0)
    for i, c in enumerate(Clusters):
        plt.scatter(TSNE_coord_clusters[0][c], TSNE_coord_clusters[1][c], c=params.colors[i], marker=params.markers[i], s=markersize[1])
    plt.scatter(TSNE_coord_clusters[0][non_indeces], TSNE_coord_clusters[1][non_indeces], c='grey', marker='o', s=markersize[0])
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig(name, dpi=100)
    plt.close() 


def plot_combined_classes (Clusters, Combined_Class):
    ratios = []
    for c in Clusters:
        comb = Combined_Class[c]
        ratios.append(len(np.where(comb=='r2')[0])/len(c))

    plt.clf()
    plt.title("r-process clusters", size=24)
    plt.xlabel ("Relative r2 content", size=24)
    plt.ylabel("Count", size=24)
    plt.tick_params(labelsize=18)
    plt.hist(ratios, color='black', linewidth=2, histtype='step', bins=20)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/combined_classes.png", dpi=100)
    plt.close() 


if __name__ == '__main__':
    file_input = params.final_sample_file
    Clusters = clusters.final_clusters
    kmeans_clusters = clusters.kmeans_clusters
    meanshift_clusters = clusters.meanshift_clusters
    afftypropagation_clusters = clusters.afftypropagation_clusters
    aggloclustering_euclidian_clusters = clusters.aggloclustering_euclidian_clusters

    Name = get_column(0, float, file_input)
    FeH = get_column(1, float, file_input)
    SrBa = get_column(3, float, file_input)
    BaEu = get_column(4, float, file_input)
    EuFe = get_column(5, float, file_input) 
    CFe = get_column(2, float, file_input)
    A_C = get_column(6, float, file_input)
    RV = get_column(7, float, file_input)
    Energy = get_column(8, float, file_input)
    L_z = get_column(9, float, file_input)
    L_p = get_column(10, float, file_input)
    I_3 = get_column(11, float, file_input)
    vRg = get_column(12, float, file_input)
    vTg = get_column(13, float, file_input)
    r_apo = get_column(14, float, file_input) 
    l = get_column(16, float, file_input)
    b = get_column(17, float, file_input)
    dist = get_column(18, float, file_input)
    Combined_Class = get_column(19, str, file_input)

    plot_data("data_F_C.png", '[Fe/H]', '[C/Fe]', FeH, CFe)
    plot_data("data_E_Lz.png", 'Energy', 'L_z', Energy, L_z)
    plot_data("data_E_Lp.png", 'Energy', 'L_p', Energy, L_p)
    plot_data("data_E_I3.png", 'Energy', 'I_3', Energy, I_3)
    plot_data("data_Lz_Lp.png", 'L_z', 'L_p', L_z, L_p)
    plot_data("data_Lz_I3.png", 'L_z', 'I_3', L_z, I_3)
    plot_data("data_Lp_I3.png", 'L_p', 'I_3', L_p, I_3)

    colors = params.colors
    markers = params.markers
    plot_concatenated_clusters("clusters_F_C.png", Clusters, colors, markers, '[Fe/H]', '[C/Fe]', FeH, CFe, True)
    plot_concatenated_clusters("clusters_F_AC.png", Clusters, colors, markers, '[Fe/H]', 'A(C)', FeH, A_C, True)
    plot_concatenated_clusters("clusters_E_I3.png", Clusters, colors, markers, 'Energy', 'I_3', Energy, I_3, False)
    plot_concatenated_clusters("clusters_rapo_CFe.png", Clusters, colors, markers, 'r_apo', '[C/Fe]', r_apo, CFe, True)
    plot_concatenated_clusters("clusters_rapo_EuFe.png", Clusters, colors, markers, 'r_apo', '[Eu/Fe]', r_apo, EuFe, True)
    plot_concatenated_clusters("clusters_rapo_SrBa.png", Clusters, colors, markers, 'r_apo', '[Sr/Ba]', r_apo, SrBa, True)
    plot_concatenated_clusters("clusters_rapo_BaEu.png", Clusters, colors, markers, 'r_apo', '[Ba/Eu]', r_apo, BaEu, True)

    plot_concatenated_clusters("clusters_1_EuFe_FeH.png", kmeans_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    plot_concatenated_clusters("clusters_2_EuFe_FeH.png", meanshift_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    plot_concatenated_clusters("clusters_3_EuFe_FeH.png", afftypropagation_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    plot_concatenated_clusters("clusters_4_EuFe_FeH.png", aggloclustering_euclidian_clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])
    plot_concatenated_clusters("clusters_all_EuFe_FeH.png", Clusters, colors, markers, '[Fe/H]', '[Eu/Fe]', FeH, EuFe, True, Xlim=[-4,-1.2], Ylim=[0,2], markersize=[50, 300])

    plot_mean_metallicities_RV (Clusters, colors, FeH, RV)
    plot_orbital_velocities (Clusters, vRg, vTg)
    plot_vTg_rapo (Clusters, vTg, r_apo)
    plot_coordinates (Clusters, l, b)
    plot_distances_RV (Clusters, dist, RV)

    plot_t_SNE (Clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE.png", markersize=[50, 300])
    plot_t_SNE (clusters.kmeans_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_Kmeans.png")
    plot_t_SNE (clusters.meanshift_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_meanshift.png")
    plot_t_SNE (clusters.afftypropagation_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_afftypropagation.png")
    plot_t_SNE (clusters.aggloclustering_euclidian_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_aggloclustering_euclidian.png")
    plot_t_SNE (clusters.aggloclustering_manhattan_clusters, [Energy, L_z, L_p, I_3], "plots/t_SNE_aggloclustering_manhattan.png")

    plot_combined_classes (Clusters, Combined_Class)
