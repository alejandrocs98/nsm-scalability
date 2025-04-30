#!/#!/bin/python3 python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import linregress, pearsonr, spearmanr, norm
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_curve, auc
import time
import distinctipy

from glob import glob
import re

import pickle as pkl
import sys
import os
import gc

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from nsm.nsm import *

params = {'legend.fontsize': 15,
          'figure.figsize': (7, 7),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'axes.linewidth': 3,
          'xtick.labelsize':18,
          'ytick.labelsize':18,
          'svg.fonttype':'none'}
plt.rcParams.update(params)
np.random.seed(12345)

mediators_ = {
    'mediators1': ['R01'],
    'mediators2': ['R09'],
    'mediators3': ['R01', 'R05'],
    'mediators4': ['R01', 'R13'],
    'mediators5': ['R01', 'R03', 'R07', 'R10', 'R13'],
    'mediators6': ['R01','R03','R05','R07','R08','R09','R10','R12','R13','R15'],
    'mediators7': ['R01','R02','R03','R04','R05','R06','R07','R08','R09','R10','R11','R12','R13','R14','R15']
}

model_paths = glob('results/parameters/parameter_*.csv')
model_paths.sort()
log_file = 'processed_parameters_jsd.txt'

for model_path in model_paths:
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
    else:
        processed_files = set()
    if model_path in processed_files:
        continue
    args = model_path.split('_')
    sps  = int(args[1].split('s')[0])
    rscs = int(args[1].split('s')[-1].split('r')[0])
    cplx = args[2]
    cs   = args[3]
    exps = int(args[4].replace('exps',''))
    md   = args[5]
    k    = int(args[6].split('k')[0])

    print(f'Processing {model_path}')

    species   = [f'S{i:02d}' for i in range(1, sps+1)]
    resources = [f'R{i:02d}' for i in range(1, rscs+1)]
    mediators = mediators_[md]

    variables = species + mediators
    md_indx = [resources.index(m) for m in mediators]
    
    param_df_summary = pd.read_csv(f'results/parameters/parameter_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{md}_{k}k.csv')
    # if param_df_summary contains NaN values or inf values, drop those rows, continue the loop and process the next file
    # if param_df_summary.isnull().values.any() or np.isinf(param_df_summary.values).any():
    if param_df_summary.isnull().values.any():
        with open('nans_parameters_.txt', 'a') as f:
            f.write(model_path + '\n')
        print(f'File {model_path} contains NaN or inf values, skipping...')
        continue

    param_df_summary_c = param_df_summary[param_df_summary['Parameter'] == 'Consumption'].copy()
    est_C_iα = param_df_summary_c.pivot_table(index='Target', columns='Source', values='Mean').reset_index()
    est_C_iα.columns.name = None
    est_C_iα.set_index('Target', inplace=True)
    est_C_iα.index.name = None
    est_C_iα.sort_index(axis=0, inplace=True)

    param_df_summary_p = param_df_summary[param_df_summary['Parameter'] == 'Production'].copy()
    est_P_iα = param_df_summary_p.pivot_table(index='Source', columns='Target', values='Mean').reset_index()
    est_P_iα.columns.name = None
    est_P_iα.set_index('Source', inplace=True)
    est_P_iα.index.name = None
    est_P_iα.sort_index(axis=0, inplace=True)

    # prep parameter files
    w = np.array([1] + 4*[0.7] + 3*[0.5] + 2*[0.7] + 2*[0.4] + 3*[0.5])
    l = np.array([0.7] * 15)

    c_iα = pd.read_csv(f'parameters/consumer_preference_{sps}s{rscs}r_{cplx}.csv', index_col=0)
    sp_map = {'S{:01d}'.format(i): 'S{:02d}'.format(i+1) for i in range(sps)}
    rcs_map = {'R{:01d}'.format(i): 'R{:02d}'.format(i+1) for i in range(rscs)}
    rpl_map = {**sp_map, **rcs_map}
    c_iα.columns = c_iα.columns.to_series().replace(rpl_map)
    c_iα.index = c_iα.index.to_series().replace(rpl_map)
    D_iαβ = np.load(f'parameters/metabolic_matrix_{sps}s{rscs}r_{cplx}.npy')
    P_iαβ = (l * w) * c_iα.values[:, np.newaxis, :] * D_iαβ

    C_iα = c_iα.loc[species, mediators]
    P_iα = pd.DataFrame((P_iαβ.sum(axis=2)/ w)[:, md_indx], columns=mediators, index=species)

    jsds = {
        'species': [], 
        'resources': [], 
        'complexity': [],
        'mediators': [], 
        'case': [], 
        'experiments': [],
        'k': [], 
        'type': [],
        'jsd': []
    }

    # Calculate Procrustes distance
    jsd_c = jensenshannon(est_C_iα.values.flatten(), C_iα.values.flatten())
    jsds['species'].append(sps)
    jsds['resources'].append(rscs)
    jsds['complexity'].append(cplx)
    jsds['mediators'].append(md)
    jsds['case'].append(cs)
    jsds['experiments'].append(exps)
    jsds['k'].append(k)
    jsds['type'].append('Consumption')
    jsds['jsd'].append(jsd_c)

    jsd_p = jensenshannon(est_P_iα.values.flatten(), P_iα.values.flatten())
    jsds['species'].append(sps)
    jsds['resources'].append(rscs)
    jsds['complexity'].append(cplx)
    jsds['mediators'].append(md)
    jsds['case'].append(cs)
    jsds['experiments'].append(exps)
    jsds['k'].append(k)
    jsds['type'].append('Production')
    jsds['jsd'].append(jsd_p)

    jsds_df = pd.DataFrame(jsds)
    jsds_df.to_csv('parameters_jsd.csv', mode='a', index=False, header=False)

    print(f'Processed {model_path}')

    # save processed file
    with open(log_file, 'a') as f:
        f.write(model_path + '\n')

    plt.clf()
    plt.cla()
    plt.close('all')

    # clean up
    del param_df_summary, param_df_summary_c, param_df_summary_p
    del est_C_iα, est_P_iα
    del jsds_df, jsds
    del c_iα, P_iα, D_iαβ, P_iαβ
    gc.collect()