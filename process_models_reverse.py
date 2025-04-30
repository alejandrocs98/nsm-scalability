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
import json
import sys
import os

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

# some functions
def get_params(self, n_sample=500):

    # parameter names
    param_names = []

    # consumption rates
    for md in self.mediators:
        for sp in self.species:
            param_names += [md+"->"+sp]

    # production rates
    for md in self.mediators:
        for sp in self.species:
            param_names += [sp+"->"+md]

    # mediator degradation rate
    for md in self.mediators:
        param_names += [f"d({md})"]

    # sample parameters
    y = np.random.randn(n_sample, self.d)
    z = batch_T(self.shapes, y, self.lmbda)

    # init vector to store parameter samples
    params = np.zeros([n_sample, 2 * self.n_m * self.n_s + self.n_m])

    # transform and reshape parameters
    for i, z_i in enumerate(z):

        # transform / reshape parameters
        C_i, P_i, d_m, *rest = transform(reshape(self.shapes, z_i))
        params[i] = np.concatenate((C_i.ravel(), P_i.ravel(), d_m))

    # compute mean / stdv
    # C_mean = np.mean(C, 0).ravel()
    # C_stdv = np.std(C, 0).ravel()
    # P_mean = np.mean(P, 0).ravel()
    # P_stdv = np.std(P, 0).ravel()

    # save to dataframe
    param_df = pd.DataFrame()
    # param_df['Parameter'] = param_names
    # param_df['$\mu$'] = np.append(C_mean, P_mean)
    # param_df['$\sigma$'] = np.append(C_stdv, P_stdv)
    for i, param_name in enumerate(param_names):
        param_df[param_name] = np.array(params[:, i], float)

    return param_df

def gaussian_exceedance_prob(means, stds, threshold):
    """
    Computes the probability that each normally distributed variable
    (characterized by means and stds) exceeds the given threshold.
    
    Parameters:
    means (ndarray): A matrix of mean values.
    stds (ndarray): A matrix of standard deviations (must be the same shape as means).
    threshold (float): The threshold value.
    
    Returns:
    ndarray: A matrix of exceedance probabilities.
    """
    return 1 - norm.cdf((threshold - means) / stds)

def get_param_stats(self):

    # get mean and stdv
    mu = self.z
    sigma = np.exp(self.lmbda[self.d:])

    # mean and standard deviation of untransformed variables (real valued)
    C_mu, P_mu, d_mu, *rest = reshape(self.shapes, mu)
    C_s, P_s, d_s, *rest = reshape(self.shapes, sigma)
    
    # acceptance probabilities 
    C_pvals = gaussian_exceedance_prob(C_mu, C_s, -3.)
    P_pvals = gaussian_exceedance_prob(P_mu, P_s, -3.)
    d_pvals = gaussian_exceedance_prob(d_mu, d_s, -3.)
    
    # actual values passed through transform
    C = g(C_mu)
    P = g(P_mu)
    d = g(d_mu)
    
    return C, C_pvals, P, P_pvals 

# Define parameter type
def parameter_type(row):
    if row['Source'].startswith('S'):
        return 'Production'
    else:
        return 'Consumption'

model_paths = glob('models/model_*.pkl')
model_paths.sort(reverse=True) 
log_file = 'processed_files.txt'

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

    # if cs == 'sparse':
    #     continue

    species   = [f'S{i:02d}' for i in range(1, sps+1)]
    resources = [f'R{i:02d}' for i in range(1, rscs+1)]
    mediators = mediators_[md]

    # load data + model
    df      = pd.read_csv(f"datasets/data_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{k}.csv")
    df_test = pd.read_csv(f"datasets/data_{sps}s{rscs}r_{cplx}_{cs}_test.csv")

    # drop treatments with zero initial biomass
    for exp_name, exp_df in df_test.groupby("Treatments"):
        if np.all(exp_df.loc[exp_df['Time']==0,exp_df.columns.str.contains('S')].isna()) or exp_df.loc[exp_df['Time']==0,exp_df.columns.str.contains('S')].values[0].sum() == 0:
            df_test = df_test[df_test.Treatments != exp_name]

    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    # run predictions
    pred_dfs = []
    for exp_name, exp_df in df_test.groupby("Treatments"):
        
        # make sure comm_data is sorted in chronological order
        exp_df.sort_values(by='Time', ascending=True, inplace=True)

        # get initial condition and evaluation times
        t_span = exp_df.Time.values
        Y_m = exp_df.loc[:,species + mediators].values

        # predict
        s_pred, s_var, m_pred, m_var = model.predict(Y_m, t_span)

        # save dataframe with predictions
        df_pred = pd.DataFrame()
        df_pred["Treatments"] = [exp_name] * len(t_span)
        df_pred['Time'] = t_span
            
        # save mean
        df_pred[[f'E[{s}]' for s in species]] = s_pred
        for s in species:
            if np.all(df_pred[f'E[{s}]'].values == df_pred[f'E[{s}]'].values[0]):
                df_pred[f'E[{s}]'] = np.nan
        df_pred[[f'E[{m}]' for m in mediators]] = m_pred
        for m in mediators:
            if np.all(df_pred[f'E[{m}]'].values == df_pred[f'E[{m}]'].values[0]):
                df_pred[f'E[{m}]'] = np.nan
            
        # save variance
        df_pred[[f'V[{s}]' for s in species]] = s_var
        for s in species:
            if np.all(df_pred[f'V[{s}]'].values == df_pred[f'V[{s}]'].values[0]):
                df_pred[f'V[{s}]'] = np.nan
        df_pred[[f'V[{m}]' for m in mediators]] = m_var
        for m in mediators:
            if np.all(df_pred[f'V[{m}]'].values == df_pred[f'V[{m}]'].values[0]):
                df_pred[f'V[{m}]'] = np.nan
        # save the dataframe
        pred_dfs.append(df_pred)
    # concatenate all dataframes
    df_pred = pd.concat(pred_dfs, ignore_index=True)
    df_pred.sort_values(by=['Treatments', 'Time'], inplace=True)
    df_pred.to_csv(f'results/predictions/predictions_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{md}_{k}k.csv', index=False)

    param_df = get_params(model)
    param_df_ = param_df.copy().loc[:, param_df.columns.str.contains("->")]
    param_df_ = param_df_.melt(var_name='Parameter', value_name='Value')
    param_df_mean = param_df_.groupby(['Parameter']).mean().reset_index()
    param_df_mean['Type'] = 'Mean'

    # Calculate standard deviation
    param_df_std = param_df_.groupby(['Parameter']).std().reset_index()
    param_df_std['Type'] = 'Std'

    # Concatenate all statistics
    param_df_ = pd.concat([param_df_mean, 
                        param_df_std, 
                        ])

    # Remove rows with NaN values
    param_df_summary = param_df_.copy().dropna()

    # Pivot the Type column to get the mean, std, median, and mode in the same row
    param_df_summary = param_df_summary.pivot_table(index=['Parameter'], columns='Type', values='Value').reset_index()
    param_df_summary.index.name = None
    param_df_summary.set_index(['Parameter'], inplace=True)
    param_df_summary.reset_index(inplace=True)

    # Split parameter column to source and target
    param_df_summary[['Source', 'Target']] = param_df_summary['Parameter'].str.split("->", expand=True)

    param_df_summary['Parameter'] = param_df_summary.apply(parameter_type, axis=1)
    param_df_summary.index.name = None

    param_df_summary_c = param_df_summary[param_df_summary['Parameter'] == 'Consumption'].copy()
    param_df_summary_p = param_df_summary[param_df_summary['Parameter'] == 'Production'].copy()

    param_df_summary.to_csv(f'results/parameters/parameter_{sps}s{rscs}r_{cplx}_{cs}_{exps}exps_{md}_{k}k.csv', index=False)

    # save processed file
    with open(log_file, 'a') as f:
        f.write(model_path + '\n')