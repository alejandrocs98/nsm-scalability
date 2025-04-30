#!/#!/bin/python3 python3
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import linregress, pearsonr, spearmanr, norm
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

model_paths = glob('results/predictions/predictions_*.csv')
model_paths.sort()
# model_paths.sort(reverse=True)
log_file = 'processed_correlations.txt'

for model_path in model_paths:
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed_files = set(line.strip() for line in f)
    else:
        processed_files = set()
    if model_path in processed_files:
        continue
    if model_path == 'results/predictions/predictions_12s15r_simple_baseline_500exps_mediators7_1k.csv':
        continue
    args = model_path.split('_')
    sps  = int(args[1].split('s')[0])
    rscs = int(args[1].split('s')[-1].split('r')[0])
    cplx = args[2]
    cs   = args[3]
    exps = int(args[4].replace('exps',''))
    md   = args[5]
    k    = int(args[6].split('k')[0])

    species   = [f'S{i:02d}' for i in range(1, sps+1)]
    resources = [f'R{i:02d}' for i in range(1, rscs+1)]
    mediators = mediators_[md]

    print(f'Processing {model_path}')

    # load data + model
    df      = pd.read_csv(f"datasets/data_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{k}.csv")
    df_test = pd.read_csv(f"datasets/data_{sps}s{rscs}r_{cplx}_{cs}_test.csv")

    # drop treatments with zero initial biomass
    for exp_name, exp_df in df_test.groupby("Treatments"):
        if np.all(exp_df.loc[exp_df['Time']==0,exp_df.columns.str.contains('S')].isna()) or exp_df.loc[exp_df['Time']==0,exp_df.columns.str.contains('S')].values[0].sum() == 0:
            df_test = df_test[df_test.Treatments != exp_name]

    with open(f'models/model_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{md}_{k}k.pkl', 'rb') as f:
        model = pkl.load(f)

    df_pred = pd.read_csv(f'results/predictions/predictions_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{md}_{k}k.csv')

    df_copy = df_test.copy()
    variables = species + mediators
    for exp_name, exp_df in df_copy.groupby("Treatments"):
        for variable in variables:
            if np.all(exp_df[variable].values == exp_df[variable].values[0]):
                df_copy.loc[df_copy.Treatments == exp_name, variable] = np.nan
        if np.all(exp_df.loc[exp_df['Time']==0,exp_df.columns.str.contains('S')].isna()) or exp_df.loc[exp_df['Time']==0,exp_df.columns.str.contains('S')].values[0].sum() == 0:
            df_copy = df_copy[df_copy.Treatments != exp_name]
    
    correlations = {
        'species': [], 
        'resources': [], 
        'complexity': [],
        'mediators': [], 
        'case': [], 
        'experiments': [],
        'k': [], 
        'variable': [],
        'type': [],
        'correlation': []
    }

    colors_sps = distinctipy.get_colors(sps, pastel_factor=0.3)
    colors_mds = distinctipy.get_colors(len(mediators), pastel_factor=0.3)
    
    fig, ax = plt.subplots()
    for i, variable in enumerate(species):
        measured_values = df_copy.loc[df_copy.Time > 0, variable].values
        measured_values = measured_values[~np.isnan(measured_values)]
        predicted_values = df_pred.loc[df_pred.Time > 0, f'E[{variable}]'].values
        predicted_values = predicted_values[~np.isnan(predicted_values)]
        predicted_errors = df_pred.loc[df_pred.Time > 0, f'V[{variable}]'].values ** 0.5
        predicted_errors = predicted_errors[~np.isnan(predicted_errors)]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(measured_values, predicted_values)

        correlations['species'].append(sps)
        correlations['resources'].append(rscs)
        correlations['complexity'].append(cplx)
        correlations['mediators'].append(md)
        correlations['case'].append(cs)
        correlations['experiments'].append(exps)
        correlations['k'].append(k)
        correlations['variable'].append(variable)
        correlations['type'].append('S')
        correlations['correlation'].append(correlation)

        # Plot the parity plot
        plt.errorbar(measured_values, predicted_values, yerr=predicted_errors, color=colors_sps[i % len(colors_sps)],
                    fmt='o', ecolor='gray', markersize=8, markeredgecolor='k', capsize=3, label=f'{variable} r={correlation:.2f}', alpha=0.8)

        # Add line of perfect agreement
        plt.plot([measured_values.min(), measured_values.max()],
             [measured_values.min(), measured_values.max()],
             linestyle='--')

        # Add legend with Pearson correlation coefficient
        # plt.legend(['Line of Perfect Agreement', f'r = {correlation:.2f}'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set labels and title
        plt.xlabel(f'Measured')
        plt.ylabel(f'Predicted')
        plt.title(f'{sps} Species | {md} | {exps} Experiments\n{cplx} Complexity | {cs} Case\nSpecies')

    fig.savefig(f'plots/correlation-validations/validation_correlation_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{md}_{k}k_species.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    for i, variable in enumerate(mediators):
        measured_values = df_copy.loc[df_copy.Time > 0, variable].values
        measured_values = measured_values[~np.isnan(measured_values)]
        predicted_values = df_pred.loc[df_pred.Time > 0, f'E[{variable}]'].values
        predicted_values = predicted_values[~np.isnan(predicted_values)]
        predicted_errors = df_pred.loc[df_pred.Time > 0, f'V[{variable}]'].values ** 0.5
        predicted_errors = predicted_errors[~np.isnan(predicted_errors)]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(measured_values, predicted_values)

        correlations['species'].append(sps)
        correlations['resources'].append(rscs)
        correlations['complexity'].append(cplx)
        correlations['mediators'].append(md)
        correlations['case'].append(cs)
        correlations['experiments'].append(exps)
        correlations['k'].append(k)
        correlations['variable'].append(variable)
        correlations['type'].append('R')
        correlations['correlation'].append(correlation)

        # Plot the parity plot
        plt.errorbar(measured_values, predicted_values, yerr=predicted_errors, color=colors_mds[i % len(colors_mds)],
                    fmt='o', ecolor='gray', markersize=8, markeredgecolor='k', capsize=3, label=f'{variable} r={correlation:.2f}', alpha=0.8)

        # Add line of perfect agreement
        plt.plot([measured_values.min(), measured_values.max()],
             [measured_values.min(), measured_values.max()],
             linestyle='--')

        # Add legend with Pearson correlation coefficient
        # plt.legend(['Line of Perfect Agreement', f'r = {correlation:.2f}'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set labels and title
        plt.xlabel(f'Measured')
        plt.ylabel(f'Predicted')
        plt.title(f'{sps} Species | {md} | {exps} Experiments\n{cplx} Complexity | {cs} Case\nMediators')

    fig.savefig(f'plots/correlation-validations/validation_correlation_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{md}_{k}k_mediators.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.clf()
    plt.cla()
    plt.close('all')

    correlations_df = pd.DataFrame(correlations)
    correlations_df.to_csv('correlations.csv', mode='a', index=False, header=False)

    # save processed file
    with open(log_file, 'a') as f:
        f.write(model_path + '\n')
    print(f'Processed {model_path}')

    # clean up
    del df, df_test, df_pred, df_copy, correlations_df, correlations, fig, ax, model
    gc.collect()