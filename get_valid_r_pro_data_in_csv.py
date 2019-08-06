from __future__ import division
import numpy as np
import csv
import params


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def return_r_pro_indeces (Combined_Class):
    return np.where(np.logical_or(Combined_Class=='r1', Combined_Class=='r2'))[0]


def return_good_data_indeces (var_list):
    indeces = range(len(var_list[0]))
    for v in var_list:
        indeces = np.intersect1d(indeces,np.where(v!=-9.999)[0])
        indeces = np.intersect1d(indeces,np.where(~np.isnan(v)))
    return indeces


def return_valid_orbit_indeces (Energy):
    return np.where(Energy<0)[0]


def return_vars_by_indeces (var_list, indeces):
    return [v[indeces] for v in var_list]


if __name__ == '__main__':
    file_input = params.input_file
    file_kin = params.kin_file
    
    Name = get_column(0, str, file_kin)
    Energy = get_column(35, float, file_kin) 
    L_z = get_column(37, float, file_kin)
    L_p = get_column(39, float, file_kin)
    I_3 = get_column(41, float, file_kin)

    FeH = get_column(14, float, file_input)
    CFe = get_column(20, float, file_input) 
    SrBa = get_column(30, float, file_input)
    BaEu = get_column(28, float, file_input)
    EuFe = get_column(26, float, file_input)   

    Combined_Class = get_column(119, str, file_input)

    RV = get_column(9, float, file_kin)
    vRg = get_column(31, float, file_kin)
    vTg = get_column(32, float, file_kin)
    r_apo = get_column(47, float, file_kin)
    ecc = get_column(45, float, file_kin)

    l = get_column(5, float, file_kin)
    b = get_column(6, float, file_kin)
    dist = get_column(10, float, file_kin)

    ra = get_column(1, float, file_kin)
    dec = get_column(2, float, file_kin)
    pmra = get_column(3, float, file_kin)
    pmdec = get_column(4, float, file_kin)

    indeces = return_r_pro_indeces(Combined_Class)
    indeces = np.intersect1d(indeces, return_good_data_indeces([Energy, L_z, L_p, I_3, FeH, CFe, SrBa, BaEu, EuFe, RV, vRg, vTg, r_apo, ecc, l, b, dist, ra, dec, pmra, pmdec]))
    indeces = np.intersect1d(indeces, return_valid_orbit_indeces(Energy))

    Name, Energy, L_z, L_p, I_3, FeH, CFe, SrBa, BaEu, EuFe, RV, vRg, vTg, r_apo, ecc, l, b, dist, Combined_Class, ra, dec, pmra, pmdec = return_vars_by_indeces([Name, Energy, L_z, L_p, I_3, FeH, CFe, SrBa, BaEu, EuFe, RV, vRg, vTg, r_apo, ecc, l, b, dist, Combined_Class, ra, dec, pmra, pmdec], indeces)

    A_C = 8.43 + CFe + FeH

    SaveFile = open(params.final_sample_file,'w')
    wr = csv.writer(SaveFile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Name']+['[Fe/H]']+['[C/Fe]']+['[Sr/Ba]']+['[Ba/Eu]']+['[Eu/Fe]']+['A_C']+['RV']+['Energy']+['Lz']+['Lp']+['I3'] + ['vRg'] + ['vTg'] + ['r_apo'] + ['ecc'] + ['l'] + ['b'] + ['dist'] + ['Combined Class'] + ['RA'] + ['DEC'] +['pmRA']+['pmDEC'])
    for i in range(len(Name)):
        wr.writerow([Name[i]]+[FeH[i]]+[CFe[i]]+[SrBa[i]]+[BaEu[i]]+[EuFe[i]]+[A_C[i]]+[RV[i]]+[Energy[i]]+[L_z[i]]+[L_p[i]]+[I_3[i]]+[vRg[i]]+[vTg[i]]+[r_apo[i]]+[ecc[i]] + [l[i]] + [b[i]] + [dist[i]] + [Combined_Class[i]] + [ra[i]] + [dec[i]] + [pmra[i]] + [pmdec[i]])
