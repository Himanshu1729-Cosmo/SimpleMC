##
##
# !!!!!THIS IS DEPRECATED!!!!!
##
# SEE SimpleCMBLikelihood.py
##
# This module implements Wang+Wang CMB
##

import sys
from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import scipy.linalg as la
import numpy as np

print("Wang Wang DEPRECATED!")
#sys.exit(1)


class WangWangCMB(BaseLikelihood):
    def __init__(self, name, mean, err, cor):
        BaseLikelihood.__init__(self, name)
        self.mean = np.array(mean)
        # wang and wang give correlation matrix
        # and errors and presumably we need to multipy
        # them back
        err = np.array(err)
        cov = np.array(cor)*np.outer(err, err)
        self.icov = la.inv(cov)

    def setTheory(self, theory):
        self.theory_ = theory
        self.theory_.setNoObh2prior()

    def loglike(self):
        delt = self.theory_.WangWangVec()-self.mean
        return -np.dot(delt, np.dot(self.icov, delt))/2.0
    
class PLKLikelihood(WangWangCMB):
    def __init__(self, matrices="WW_PLK18_ΛCDM"):
        if matrices == "WW_WMAP7":
            # arXiv:1304.4514 (Planck + lensing + WP see Eq~12 and 13)
            mean = [301.57, 1.7407, 0.02228]
            err = [0.18, 0.0094, 0.00030]
            cov = [[1.0, 0.5250, -0.4235],
                   [0.5250, 1.0, -0.6925],
                   [-0.4235, -0.6925,  1.0]]
            
            # arXiv:1304.4514 (WMAP9 see Eq~14 and 15)
        elif matrices == "WW_WMAP9":
            # base_omegak
            mean = [302.02, 1.7327, 0.02260]
            err = [0.66, 0.0164, 0.00053]
            cov = [[1.0, 0.3883, -0.6089],
                   [0.3883, 1.0, -0.5239],
                   [-0.6089, -0.5239,  1.0]]

        elif matrices == "PLA1":
            # base_omegak
            mean = [3.01510344e+02,   1.74340204e+00,   2.23128506e-02]
            err = [0.18550629,  0.00928648,  0.00031125]
            cov = [[1.,          0.57063287, -0.48438085],
                   [0.57063287,  1.,         -0.67948316],
                   [-0.48438085, -0.67948316,  1.]]
            
        elif matrices == "PLA2":
            # base_nrun_r_omegak
            mean = [3.01412183e+02, 1.73538851e+00, 2.27716863e-02]
            err = [0.20675659,  0.01203558,  0.00044825]
            cov = [[1.,          0.66889481, -0.59554585],
                   [0.66889481,  1.,         -0.78180126],
                   [-0.59554585, -0.78180126,  1.]]
            
        elif matrices == "PLA3":
            # base_nrun_r_omegak
            mean = [3.01539311e+02,   1.74066358e+00,   2.25879526e-02]
            err = [0.19676676,  0.01056994,  0.00037733]
            cov = [[1.,  0.62040041, -0.53470863],
                   [0.62040041,  1., -0.71007217],
                   [-0.53470863, -0.71007217,  1.]]
            
        elif matrices == 'WW_PLK15_LCDM':
            # ΛCDM arXiv:1509.00969 (Calibrated with Planck 15 TT,TE,EE + lowE)
            mean = [3.01505e+02, 1.7496, 2.2225e-02]
            err  = [0.092, 0.0050, 0.00016]
            cov  = [[1.0, 0.49, -0.37],
                    [0.49, 1.0, -0.69],
                    [-0.37, -0.69, 1.0]]
            
        elif matrices == 'WW_PLK15_wCDM':
            # wCDM arXiv:1509.00969 (Calibrated with Planck 15 TT,TE,EE + lowE)
            mean = [3.01498e+02, 1.7488, 2.228e-02]
            err  = [0.091, 0.0049, 0.00016]
            cov  = [[1.0, 0.49, -0.38],
                    [0.49, 1.0, -0.68],
                    [-0.38, -0.68, 1.0]]

        elif matrices == 'WW_PLK15_LCDM_OK':
            # ΛCDM + Ω_k arXiv:1509.00969 (Calibrated with Planck 15 TT,TE,EE + lowE)
            mean = [3.01465e+02, 1.7449, 2.241e-02]
            err  = [0.093, 0.0052, 0.00017]
            cov  = [[1.0, 0.47, -0.37],
                    [0.47, 1.0, -0.71],
                    [-0.37, -0.71, 1.0]]
            
        elif matrices == 'WW_PLK15_LCDM_AL':
            # ΛCDM + A_L arXiv:1509.00969 (Calibrated with Planck 15 TT,TE,EE + lowE)
            mean = [3.01460e+02, 1.7448, 2.240e-02]
            err  = [0.094, 0.0054, 0.00017]
            cov  = [[1.0, 0.53, -0.42],
                    [0.53, 1.0, -0.73],
                    [-0.42, -0.73, 1.0]]
            
        elif matrices == 'WW_PLK18_ΛCDM':
            # ΛCDM arXiv:1808.05724 (Calibrated with Planck 18 TT,TE,EE + lowE )
            mean = [3.01471e+02, 1.7502, 2.236e-02]
            err = [0.089, 0.0046, 0.00015]
            cov = [[1.0, 0.46, -0.33],
                   [0.46, 1.0, -0.66],
                   [-0.33, -0.66, 1.0]]
            
        elif matrices == 'WW_PLK18_wCDM':
            # wCDM arXiv:1808.05724 (Calibrated with Planck 18 TT,TE,EE + lowE )
            mean = [3.01462e+02, 1.7493, 2.239e-02]
            err  = [0.089, 0.0046, 0.00015]
            cov  = [[1.0,   0.47,  -0.34],
                    [0.47,  1.0,   -0.66],
                    [-0.34, -0.66,  1.0]]
            
        elif matrices == 'WW_PLK18_ΛCDM_OK':
            # ΛCDM + Ω_k  arXiv:1808.05724 (Calibrated with Planck 18 TT,TE,EE + lowE )
            mean = [3.01409e+02, 1.7429, 2.260e-02]
            err  = [0.091, 0.0051, 0.00017]
            cov  = [[1.0,     0.54,   -0.42],
                    [0.54,    1.0,    -0.75],
                    [-0.42,  -0.75,    1.0]]
            
        elif matrices == 'WW_PLK18_LCDM_AL':
            # ΛCDM + A_L  arXiv:1808.05724 (Calibrated with Planck 18 TT,TE,EE + lowE )
            mean = [3.01406e+02, 1.7428, 2.259e-02]
            err  = [0.090, 0.0053, 0.00017]
            cov  = [[1.0,   0.52,  -0.41],
                    [0.52,  1.0,   -0.72],
                    [-0.41, -0.72,  1.0]]

        elif matrices == "PLA":
            mean = [3.01969812e+02,   1.73081611e+00,   2.26651563e-02]
            err = [6.61780337e-01,   1.60432146e-02,   5.18589026e-04]
            cov = [[1., 0.37679434, -0.61134328],
                   [0.37679434, 1., -0.51194784],
                   [-0.61134328, -0.51194784, 1.]]
        else:
            print("Bad matrices param")
            sys.exit(1)

        WangWangCMB.__init__(self, "CMB_WW_"+matrices, mean, err, cov)
