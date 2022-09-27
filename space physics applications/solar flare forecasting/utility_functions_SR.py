# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:41:56 2021

@author: savvra
"""

import numpy as np
from scipy.stats import gaussian_kde


def normalizeRevisted(data,traindataset,axis=0): #Custom Normalize Routine
    mean = np.mean(traindataset,axis)
    std  = np.std(traindataset,axis)
    return (data-mean)/std, mean, std


def samplestrat(df, stratifying_column_name, num_to_sample, maxrows_to_est = 10000, bw_per_range = 50, eval_points = 1000 ):
    '''Take a sample of dataframe df stratified by stratifying_column_name
    '''
    strat_col_values = df[stratifying_column_name].values
    samplcol = (df.sample(maxrows_to_est)  if df.shape[0] > maxrows_to_est else df  )[stratifying_column_name].values
    vmin, vmax = min(samplcol), max(samplcol)
    pts = np.linspace(vmin,vmax  ,eval_points) 
    kernel = gaussian_kde( samplcol , bw_method = float(  (vmax - vmin)/bw_per_range  )   )
    density_estim_full = np.interp(strat_col_values, pts , kernel.evaluate(pts) )
    return df.sample(n=num_to_sample, weights = 1/(density_estim_full))