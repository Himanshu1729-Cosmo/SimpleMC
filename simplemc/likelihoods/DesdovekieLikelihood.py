from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
import numpy as np
import scipy.linalg as la
from scipy.interpolate import interp1d
from simplemc.setup_logger import cdir
import pandas as pd

class DesdovekieLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, cov_filename, ninterp=150):
        """
        Likelihood for full DES 5YR Dovekie compilation.
        """

        self.name_ = name
        BaseLikelihood.__init__(self, name)
        print("Loading", values_filename)
        da = pd.read_csv(values_filename)
        self.zcmb = da['zHD']
        self.zhelio = da['zHEL']
        self.mag = da['MU']
        self.N = len(self.mag)
        print("Loading inverse covariance", cov_filename)
        inv_cov = self._read_inv_covmat(cov_filename)
        self.icov = inv_cov
        self.cov = la.inv(inv_cov)
        # Diagonal BEFORE marginalising constant (this matches DESY5)
        self.xdiag = 1.0 / self.cov.diagonal()
        self.zmin  = self.zcmb.min()
        self.zmax  = self.zcmb.max()
        self.zmaxi = 1.1  # interpolate to 1.1, beyond that exact calc
        print("Dovekie: zmin=%f zmax=%f N=%i" % (self.zmin, self.zmax, self.N))
        self.zinter = np.linspace(1e-3, self.zmaxi, ninterp)
        
    # --------------------------------------------------
    #   Read DES-Dovekie compressed inverse covariance
    # --------------------------------------------------
    def _read_inv_covmat(self, filename):
        d = np.load(filename)
        # first array holds n
        n = d[d.files[0]][0]
        inv_cov = np.zeros((n, n))
        # second array is upper triangle of inv_cov
        inv_cov[np.triu_indices(n)] = d[d.files[1]]
        # reflect to lower triangle to make symmetric
        i_lower = np.tril_indices(n, -1)
        inv_cov[i_lower] = inv_cov.T[i_lower]

        return inv_cov

    def loglike(self):
        # interpolate distance modulus
        dist = interp1d(self.zinter,[self.theory_.distance_modulus(z) for z in self.zinter],
                        kind='cubic',bounds_error=False)(self.zcmb)
        who = np.where(self.zcmb > self.zmaxi)
        dist[who] = np.array([self.theory_.distance_modulus(z) for z in self.zcmb.loc[who]])

        tvec = self.mag.values - dist
        tvec -= (tvec * self.xdiag).sum() / self.xdiag.sum()
        chi2 = np.einsum('i,ij,j', tvec, self.icov, tvec)
        return -chi2/2
    
class Desdovekie(DesdovekieLikelihood):
    """
    Likelihood for full DES 5YR Dovekie compilation.
    """
    def __init__(self):
        DesdovekieLikelihood.__init__(self, "Desdovekie", cdir + "/data/DES-Dovekie_HD.csv",
                                        cdir + "/data/STAT+SYS.npz")
