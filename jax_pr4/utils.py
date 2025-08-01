"""
Utility functions for Planck PR4 likelihoods

This module contains helper functions used by the CMB likelihoods,
including lognormal likelihood calculations and binning utilities.
"""

from .module import *
import numpy as np
import os

# Optional h5py import
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

def lognormal(x, mu, sig, loc=0):
    """
    Compute lognormal probability density function.
    
    Parameters:
    -----------
    x : array-like
        Input values
    mu : array-like
        Mean of the log of the distribution
    sig : array-like
        Standard deviation of the log of the distribution
    loc : array-like, optional
        Location parameter (default: 0)
    
    Returns:
    --------
    array-like
        Lognormal PDF values
    """
    LN = (
        1
        / ((x - loc) * sig * sqrt(2 * pi))
        * exp(-((log(x - loc) - mu) ** 2) / (2 * sig**2))
    )
    return LN

def get_binned_D_from_theory_Cls(ell, Cl, lmin_list, lmax_list):
    """
    Convert from C_l to D_l = l(l+1)C_l/(2π), average in bins, convert back to C_l.
    
    Parameters:
    -----------
    ell : array-like
        Multipole values
    Cl : array-like
        Power spectrum values (batch, ell)
    lmin_list : array-like
        Minimum multipoles for each bin
    lmax_list : array-like
        Maximum multipoles for each bin
    
    Returns:
    --------
    array-like
        Binned D_l values (batch, n_bins)
    """
    D_fac = ell * (ell + 1) / (2 * pi)
    Dl = D_fac * Cl

    Dl_bin = array(zeros((Cl.shape[0], len(lmin_list))))

    for i, lmin in enumerate(lmin_list):
        lmax = lmax_list[i]

        if get_jax_enabled():
            Dl_bin = Dl_bin.at[:, i].set(
                mean(Dl[:, lmin.astype(int) - 2 : lmax.astype(int) - 2 + 1], axis=1)
            )
        else:
            Dl_bin[:, i] = mean(Dl[:, int(lmin) - 2 : int(lmax) - 2 + 1], axis=1)

    return Dl_bin

def get_binned_D_from_theory_Cls_tt(ell, Cl):
    """
    Convert from C_l to D_l for TT spectrum with fixed binning (2-15, 16-29).
    
    Parameters:
    -----------
    ell : array-like
        Multipole values
    Cl : array-like
        TT power spectrum values (batch, ell)
    
    Returns:
    --------
    array-like
        Binned D_l values (batch, 2)
    """
    lmin_list = [2, 16]
    lmax_list = [15, 29]

    D_fac = ell * (ell + 1) / (2 * pi)
    Dl = D_fac * Cl

    Dl_bin = array(zeros((Cl.shape[0], len(lmin_list))))

    for i, lmin in enumerate(lmin_list):
        lmax = lmax_list[i]

        if get_jax_enabled():
            Dl_bin = Dl_bin.at[:, i].set(mean(Dl[:, lmin - 2 : lmax - 2 + 1], axis=1))
        else:
            Dl_bin[:, i] = mean(Dl[:, int(lmin) - 2 : int(lmax) - 2 + 1], axis=1)

    return Dl_bin

def planck_lowE_binned_loglike(
    Cl_theory, mu_LN_EE, sig_LN_EE, loc_LN_EE, lmin_list_EE, lmax_list_EE
):
    """
    Compute Planck low-ell EE lognormal likelihood.
    
    Parameters:
    -----------
    Cl_theory : array-like
        Theoretical EE power spectrum from ell=2-30 (batch, 29)
    mu_LN_EE : array-like
        Mean parameters for lognormal bins
    sig_LN_EE : array-like
        Standard deviation parameters for lognormal bins
    loc_LN_EE : array-like
        Location parameters for lognormal bins
    lmin_list_EE : array-like
        Minimum multipoles for each bin
    lmax_list_EE : array-like
        Maximum multipoles for each bin
    
    Returns:
    --------
    array-like
        Log-likelihood values (batch,)
    """
    ell = arange(2, 30)
    Dl_theory_bin = get_binned_D_from_theory_Cls(
        ell, Cl_theory, lmin_list_EE, lmax_list_EE
    )
    loglike = array(zeros((Cl_theory.shape[0],), dtype=np.float64))

    for i in range(Dl_theory_bin.shape[1]):
        D = Dl_theory_bin[:, i]
        like_real = lognormal(D, mu_LN_EE[i], sig_LN_EE[i], loc_LN_EE[i])
        loglike += log(like_real)
    return loglike

def planck_lowT_binned_loglike(
    Cl_theory, mu_LN_TT, sig_LN_TT, lmin_list_TT, lmax_list_TT
):
    """
    Compute Planck low-ell TT lognormal likelihood.
    
    Parameters:
    -----------
    Cl_theory : array-like
        Theoretical TT power spectrum from ell=2-30 (batch, 29)
    mu_LN_TT : array-like
        Mean parameters for lognormal bins
    sig_LN_TT : array-like
        Standard deviation parameters for lognormal bins
    lmin_list_TT : array-like
        Minimum multipoles for each bin
    lmax_list_TT : array-like
        Maximum multipoles for each bin
    
    Returns:
    --------
    array-like
        Log-likelihood values (batch,)
    """
    ell = arange(2, 30)
    Dl_theory_bin = get_binned_D_from_theory_Cls_tt(ell, Cl_theory)
    loglike = array(zeros((Cl_theory.shape[0],), dtype=np.float64))
    
    for i in range(Dl_theory_bin.shape[1]):
        D = Dl_theory_bin[:, i]
        like_real = lognormal(D, mu_LN_TT[i], sig_LN_TT[i])
        loglike += log(like_real)
    return loglike

def load_lognormal_data(data_dir, filename):
    """
    Load lognormal fit data from file.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data file
    filename : str
        Name of the data file
    
    Returns:
    --------
    tuple
        Unpacked data from the file
    """
    file_path = os.path.join(data_dir, filename)
    
    if get_jax_enabled():
        # For JAX, read as regular numpy first then convert
        data = []
        with open(file_path, "r") as file:
            next(file)  # skip header line
            for line in file:
                line_data = [float(item) for item in line.split()]
                data.append(line_data)
        
        # Transpose data to unpack correctly
        data_transposed = list(zip(*data))
        
        # Convert to JAX arrays
        result = [array(x) for x in data_transposed]
        return result
    else:
        # For NumPy, use loadtxt directly
        return np.loadtxt(file_path, unpack=True)

def safe_inv(matrix, rcond=1e-9):
    """
    Safely compute matrix inverse with condition number check.
    
    Parameters:
    -----------
    matrix : array-like
        Input matrix
    rcond : float
        Reciprocal condition number threshold
    
    Returns:
    --------
    array-like
        Inverse matrix
    """
    if get_jax_enabled():
        return linalg.pinv(matrix, rcond=rcond)
    else:
        return linalg.inv(matrix)