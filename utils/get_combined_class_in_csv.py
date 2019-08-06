from __future__ import division
import numpy as np
import csv
import params


def get_column(N, Type, filename):
    return np.genfromtxt(filename, delimiter=',', dtype=Type, skip_header=1, usecols=N, comments=None)


def return_good_data_indeces (var_list):
    indeces = range(len(var_list[0]))
    for v in var_list:
        indeces = np.intersect1d(indeces,np.where(v!=-9.999)[0])
        indeces = np.intersect1d(indeces,np.where(~np.isnan(v)))
    return indeces


if __name__ == '__main__':
    file_input = params.input_file
    file_kin = params.kin_file

    FeH = get_column(14, float, file_input)
    SrBa = get_column(30, float, file_input)
    BaEu = get_column(28, float, file_input)
    EuFe = get_column(26, float, file_input)

    Combined_Class = ['' for i in range(len(FeH))]
    indeces = return_good_data_indeces ([FeH, SrBa, BaEu, EuFe])

    for i in indeces:
        if (EuFe[i]>1 and BaEu[i]<0):
            Combined_Class[i] = 'r2'
        elif (EuFe[i]>=0.3 and EuFe[i]<=1 and BaEu[i]<0):
            Combined_Class[i] = 'r1'
        elif (EuFe[i]<0.3 and SrBa[i]>0.5):
            Combined_Class[i] = 'limited-r'

    SaveFile = open('combined_class.csv','w')
    wr = csv.writer(SaveFile, quoting=csv.QUOTE_NONE)
    wr.writerow(['Combined Class'] + ['placeholder'])
    for c in Combined_Class:
        wr.writerow([c] + [''])
