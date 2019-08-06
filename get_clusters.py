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
import params


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def clustering (method, Energy, L_z, L_p, I_3):
    method.fit(np.transpose(np.array([Energy, L_z, L_p, I_3])))
    labels = set(method.labels_)
    if -1 in labels:
        labels.remove(-1)
    method_cluster_indeces=[]
    print labels
    for l in labels:
        indeces = np.where(method.labels_==l)[0]
        method_cluster_indeces.append(indeces)
    return method_cluster_indeces


def get_concatenated_clusters (cluster_arrays):
    clusters = [[] for i in range(len(cluster_arrays))]
    for i, cluster in enumerate(cluster_arrays):
        for c in cluster:
            for pair in list(itertools.combinations(c,2)):
                clusters[i].append(pair)
    # clusters is now a list of lists of pairs (tuples).
    all_pairs = set.intersection(*map(set,clusters))
    all_pairs = [list(s) for s in all_pairs]
    # Now the clusters will be assembled.
    found = True
    while (found):
        found = False
        for i in range(len(all_pairs)):
            for j in range(i, len(all_pairs)):
                if i!=j:
                    if bool(set(all_pairs[i]) & set(all_pairs[j])):
                        found = True
                        all_pairs[i] = list(set(all_pairs[i]+all_pairs[j]))
                        all_pairs[j] = []
        all_pairs = [l for l in all_pairs if l != []]
    return all_pairs


if __name__ == '__main__':
    file_input = params.final_sample_file
     
    Energy = get_column(5, float, file_input)
    L_z = get_column(6, float, file_input)
    L_p = get_column(7, float, file_input)
    I_3 = get_column(8, float, file_input)

    e = (Energy - np.mean(Energy))/np.std(Energy)
    lz = (L_z - np.mean(L_z))/np.std(L_z)
    lp = (L_p - np.mean(L_p))/np.std(L_p)
    i3 = (I_3 - np.mean(I_3))/np.std(I_3)

    kmeans = KMeans(n_clusters=60, algorithm="full", max_iter=2000)
    kmeans_cluster_indeces = clustering(kmeans, e, lz, lp, i3)
    kmeans_cluster_indeces = [list(c) for c in kmeans_cluster_indeces]
    cut_kmeans_cluster_indeces = [list(c) for c in kmeans_cluster_indeces if len(c)>2]

    meanshift = MeanShift(cluster_all=False, bandwidth=0.7)
    meanshift_cluster_indeces = clustering(meanshift, e, lz, lp, i3)
    meanshift_cluster_indeces = [list(c) for c in meanshift_cluster_indeces]
    cut_meanshift_cluster_indeces = [list(c) for c in meanshift_cluster_indeces if len(c)>2]

    afftypropagation = AffinityPropagation(damping=0.55)
    afftypropagation_cluster_indeces = clustering(afftypropagation, e, lz, lp, i3)
    afftypropagation_cluster_indeces = [list(c) for c in afftypropagation_cluster_indeces]
    cut_afftypropagation_cluster_indeces = [list(c) for c in afftypropagation_cluster_indeces if len(c)>2]

    aggloclustering_euclidian = AgglomerativeClustering(n_clusters=60, affinity='euclidean', linkage='ward')
    aggloclustering_euclidian_cluster_indeces = clustering(aggloclustering_euclidian, e, lz, lp, i3)
    aggloclustering_euclidian_cluster_indeces = [list(c) for c in aggloclustering_euclidian_cluster_indeces]
    cut_aggloclustering_euclidian_cluster_indeces = [list(c) for c in aggloclustering_euclidian_cluster_indeces if len(c)>2]

    aggloclustering_manhattan = AgglomerativeClustering(n_clusters=60, affinity='manhattan', linkage='single')
    aggloclustering_manhattan_cluster_indeces = clustering(aggloclustering_manhattan, e, lz, lp, i3)
    aggloclustering_manhattan_cluster_indeces = [list(c) for c in aggloclustering_manhattan_cluster_indeces]
    cut_aggloclustering_manhattan_cluster_indeces = [list(c) for c in aggloclustering_manhattan_cluster_indeces if len(c)>2]

    clusters = get_concatenated_clusters ([kmeans_cluster_indeces, meanshift_cluster_indeces, afftypropagation_cluster_indeces, aggloclustering_euclidian_cluster_indeces])
    # Only clusters with more than 2 members are left.
    cut_clusters = [c for c in clusters if len(c)>2]
    print "Clusters:"
    for c in cut_clusters:
        print c

    if os.path.isfile(params.clusters_file):
        os.remove(params.clusters_file)
    with open(params.clusters_file,'w') as clusters_file:
        clusters_file.write("from __future__ import division\n")
        clusters_file.write("import numpy as np\n")
        clusters_file.write("\n")
        clusters_file.write("\n")
        clusters_file.write("full_kmeans_clusters = " + str(list(kmeans_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("full_meanshift_clusters = " + str(list(meanshift_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("full_afftypropagation_clusters = " + str(list(afftypropagation_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("full_aggloclustering_euclidian_clusters = " + str(list(aggloclustering_euclidian_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("full_aggloclustering_manhattan_clusters = " + str(list(aggloclustering_manhattan_cluster_indeces)) + "\n")
        clusters_file.write("full_final_clusters = " + str(list(clusters)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("\n")
        clusters_file.write("kmeans_clusters = " + str(list(cut_kmeans_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("meanshift_clusters = " + str(list(cut_meanshift_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("afftypropagation_clusters = " + str(list(cut_afftypropagation_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("aggloclustering_euclidian_clusters = " + str(list(cut_aggloclustering_euclidian_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("aggloclustering_manhattan_clusters = " + str(list(cut_aggloclustering_manhattan_cluster_indeces)) + "\n")
        clusters_file.write("\n")
        clusters_file.write("final_clusters = " + str(list(cut_clusters)) + "\n")

