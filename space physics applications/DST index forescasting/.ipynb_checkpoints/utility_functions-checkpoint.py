# -*- coding: utf-8 -*-
"""
@author: Savvas Raptis

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

    
def datafrompd(dt, nt, T1, rawdata, mask, incols, outcols):
    T0 = dt * nt
    tau = T0 + T1
    m = rawdata.shape[0] - tau
    Y = np.array(rawdata.loc[tau:, outcols])
    X = np.array([rawdata.loc[i:i + T0:dt, incols].values.flatten() for i in range(m)])

    Ymask = np.array(mask.loc[tau:, outcols])
    Xmask = np.array([mask.loc[i:i + T0:dt, incols].values.flatten() for i in range(m)])
    mask = np.array([Xmask.all(axis=1) & Ymask.all(axis=1)]).T

    m = Y.shape[0]
    Y = np.array([Y[i, :] for i in range(m) if mask[i]])
    X = np.array([X[i, :] for i in range(m) if mask[i]])

    return X, Y, m