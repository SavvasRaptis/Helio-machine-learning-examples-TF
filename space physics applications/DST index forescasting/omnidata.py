#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

headers = ['year',
           'day',
           'hour',
           'Bartels',
           'IMF_spacecraft',
           'plasma_spacecraft',
           'IMF_av_npoints',
           'plasma_av_npoints',
           'av_|B|',
           '|av_B|',
           'lat_av_B_GSE',
           'lon_av_B_GSE',
           'Bx',
           'By_GSE',
           'Bz_GSE',
           'By_GSM',
           'Bz_GSM',
           'sigma_|B|',
           'sigma_B',
           'sigma_Bx',
           'sigma_By',
           'sigma_Bz',
           'Tp',
           'Np',
           'V_plasma',
           'phi_V_angle',
           'theta_V_angle',
           'Na/Np',
           'P_dyn',
           'sigma_Tp',
           'sigma_Np',
           'sigma_V',
           'sigma_phi_V',
           'sigma_theta_V',
           'sigma_Na/Np',
           'E',           
           'beta',
           'Ma',
           'Kp',
           'R',
           'Dst',
           'AE',
           'p_flux_>1MeV',
           'p_flux_>2MeV',
           'p_flux_>4MeV',
           'p_flux_>10MeV',
           'p_flux_>30MeV',
           'p_flux_>60MeV',
           'flag',
           'Ap',
           'f10.7',
           'PC',
           'AL',
           'AU',
           'M_ms']

nulls = [
        None,
        None,
        None,
        9999,
        0,
        0,
        999,
        999,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        999.9,
        9999999.,
        999.9,
        9999.,
        999.9,
        999.9,
        9.999,
        99.99,
        9999999.,
        999.9,
        9999.,
        999.9,
        999.9,
        9.999,
        999.99,
        999.99,
        999.9,
        99,
        999,
        99999,
        9999,
        999999.99,
        99999.99,
        99999.99,
        99999.99,
        99999.99,
        99999.99,
        0,
        999,
        999.9,
        999.9,
        99999,
        99999,
        99.9]

def read(file):
    odata = pd.read_table(file,header=None,names=headers,delim_whitespace=True,index_col=False)
    mask = (odata!=nulls)
    return odata,mask

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