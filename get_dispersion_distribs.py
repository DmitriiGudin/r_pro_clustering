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


min_cluster_size = params.dispersion_cluster_size[0]
max_cluster_size = params.dispersion_cluster_size[1]
N_samples = 20000 # A multiple of 1000, preferably >=10000. Otherwise, the sigmas may not be calculated properly.
bandwidth_refinement = 5


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def plot_stuff(X, Y, Y_kde, rv):
    plt.clf()
    plt.title("Gaussian distribution - KDE fit", size=24)
    plt.xlabel('X', size=24)
    plt.ylabel('G(X)', size=24)
    plt.tick_params(labelsize=18)
    plt.hist(Y, color='black', linewidth=2, density=True, histtype='step', bins=40)
    plt.plot(X, Y_kde, color='red', linewidth=2)
    plt.plot(X, rv.pdf(X), color='blue', linewidth=2)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig('test.png', dpi=100)
    plt.close()


def plot_dispersion_distrib(dispersion_arrays, var_names, var_labels):
    for i, a in enumerate(dispersion_arrays):
        x = np.linspace(min(a), max(a), N_samples)
        kde = KernelDensity(kernel='gaussian', bandwidth=(max(a)-min(a))/100/bandwidth_refinement).fit(a[:, None])
        a_kde = kde.score_samples(x[:, None])
        a_kde = np.exp(a_kde)

        plt.clf()
        plt.title("Dispersion distribution, " + str(i+min_cluster_size) + " stars", size=24)

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

        plt.gcf().set_size_inches(25.6, 14.4)
        
        filename = params.dispersion_distrib_plot_file_mask[0] + var_names[0]
        if len(var_names)>1:
            for v in var_names[1:]:
                filename += ('_' + v)
        filename += ('_'+str(i+min_cluster_size))
        filename += params.dispersion_distrib_plot_file_mask[1]
        plt.gcf().savefig(filename, dpi=100)
        plt.close()


def get_1D_dispersion_distrib(var, var_name, var_label):
    f = h5py.File(params.dispersion_distrib_file_mask[0]+var_name+params.dispersion_distrib_file_mask[1],'w')
    f.create_dataset("/dispersion", (max_cluster_size-min_cluster_size+1, N_samples), dtype='f')
    for i in range(min_cluster_size, max_cluster_size+1):
        dispersion_array = np.zeros((N_samples,))
        for j in range(N_samples):
            random.shuffle(var)
            v = var[0:i]
            dispersion_array[j] = np.std(v)
        f["/dispersion"][i-min_cluster_size] = dispersion_array[:]

    plot_dispersion_distrib (f["/dispersion"][:], [var_name], [var_label])
    f.close()
            
        
def get_2D_dispersion_distrib (Vars, var_names, var_labels):
    filename = params.dispersion_distrib_file_mask[0]+var_names[0]
    for v in var_names[1:]:
        filename += ('_'+v)
    filename += params.dispersion_distrib_file_mask[1]
    f = h5py.File(filename,'w')
    f.create_dataset("/dispersion", (max_cluster_size-min_cluster_size+1, N_samples), dtype='f')
    for i in range(len(Vars)):
        Vars[i] = (Vars[i] - np.mean(Vars[i])) / np.std(Vars[i])
    Vars = np.transpose(Vars)
    for i in range(min_cluster_size, max_cluster_size+1):
        dispersion_array = np.zeros((N_samples,))
        for j in range(N_samples):
            indeces = np.arange(0, len(Vars))
            random.shuffle(indeces)
            v = Vars[indeces[0:i]]
            dispersion_array[j] = np.std(v)
        f["/dispersion"][i-min_cluster_size] = dispersion_array[:]

    plot_dispersion_distrib (f["/dispersion"][:], var_names, var_labels)
    f.close()


if __name__ == '__main__':
    file_input = params.final_sample_file
    FeH = get_column(1, float, file_input)
    CFe = get_column(2, float, file_input)
    A_C = get_column(6, float, file_input) 
    vRg = get_column(9, float, file_input)
    vTg = get_column(10, float, file_input)
    r_apo = get_column(11, float, file_input)  
    SrBa = get_column(3, float, file_input)
    BaEu = get_column(4, float, file_input)
    EuFe = get_column(5, float, file_input)     

    get_1D_dispersion_distrib(FeH, 'FeH', '[Fe/H]')
    get_1D_dispersion_distrib(A_C, 'AC', 'A(C)')
    get_1D_dispersion_distrib(CFe, 'CFe', '[C/Fe]')
    get_2D_dispersion_distrib([FeH,A_C], ['FeH','AC'], ['[Fe/H]', 'A(C)'])
    get_1D_dispersion_distrib(vRg, 'vRg', 'vRg')
    get_1D_dispersion_distrib(vTg, 'vTg', 'vTg')
    get_1D_dispersion_distrib(r_apo, 'r_apo', 'Apocentric distance')
    get_1D_dispersion_distrib(SrBa, 'SrBa', '[Sr/Ba]')
    get_1D_dispersion_distrib(BaEu, 'BaEu', '[Ba/Eu]')
    get_1D_dispersion_distrib(EuFe, 'EuFe', '[Eu/Fe]')

    
