"""
Tools for Lollipop likelihood processing.

This module contains utility functions for the Lollipop low-ell likelihood,
including binning, data reading, covariance handling, and matrix operations.
"""

from jax_pr4.module import *
import numpy as np


class Bins(object):
    """
    lmins : list of integers
        Lower bound of the bins
    lmaxs : list of integers
        Upper bound of the bins
    """

    def __init__(self, lmins, lmaxs):
        if not (len(lmins) == len(lmaxs)):
            raise ValueError("Incoherent inputs")

        lmins = asarray(lmins)
        lmaxs = asarray(lmaxs)
        # cutfirst = logical_and(lmaxs >= 2, lmins >= 2)

        cutfirst = 29
        # hardcoded for jax jit compatibility- please fix this
        self.lmins = lmins[:cutfirst]
        self.lmaxs = lmaxs[:cutfirst]

        self.lmin = module_min(lmins[:cutfirst])
        self.lmax = module_max(lmaxs[:cutfirst])

        self.nbins = len(lmins[:cutfirst])
        self.lbin = (lmins[:cutfirst] + lmaxs[:cutfirst]) / 2.0
        self.dl = lmaxs[:cutfirst] - lmins[:cutfirst] + 1

        print("NBINS", self.nbins)
        print("LMAX", self.lmax)

        # self._derive_ext()

    @classmethod
    def fromdeltal(cls, lmin, lmax, delta_ell):
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell - 1
        return cls(lmins, lmaxs)

    # def _derive_ext(self):
    #     # for l1, l2 in zip(self.lmins, self.lmaxs):
    #     #     if l1 > l2:
    #     #         raise ValueError("Incoherent inputs")
    #     self.lmin = min(self.lmins)
    #     self.lmax = max(self.lmaxs)
    #     # if self.lmin < 1:
    #     #     raise ValueError("Input lmin is less than 1.")
    #     # if self.lmax < self.lmin:
    #     #     raise ValueError("Input lmax is less than lmin.")

    #     self.nbins = len(self.lmins)
    #     self.lbin = (self.lmins + self.lmaxs) / 2.0
    #     self.dl = self.lmaxs - self.lmins + 1

    def bins(self):
        return (self.lmins, self.lmaxs)

    def cut_binning(self, lmin, lmax):
        sel = where((self.lmins >= lmin) & (self.lmaxs <= lmax))[0]
        self.lmins = self.lmins[sel]
        self.lmaxs = self.lmaxs[sel]
        self._derive_ext()

    def _bin_operators(self, lmins, lmaxs, Dl=False, cov=False):
        nbins = 29
        lmax = 30
        if Dl:
            ell2 = arange(lmax + 1)
            ell2 = ell2 * (ell2 + 1) / (2 * pi)
        else:
            ell2 = ones(lmax + 1)

        p = array(zeros((nbins, lmax + 1)))
        q = array(zeros((lmax + 1, nbins)))

        for b, (a, z) in enumerate(zip(lmins, lmaxs)):
            dl = z - a + 1

            if get_jax_enabled():
                p = p.at[b, a : z + 1].set(ell2[a : z + 1] / dl)
                if cov:
                    q = q.at[a : z + 1, b].set(1 / ell2[a : z + 1] / dl)
                else:
                    q = q.at[a : z + 1, b].set(1 / ell2[a : z + 1])

            else:
                p[b, a : z + 1] = ell2[a : z + 1] / dl
                if cov:
                    q[a : z + 1, b] = 1 / ell2[a : z + 1] / dl
                else:
                    q[a : z + 1, b] = 1 / ell2[a : z + 1]

        return p, q

    def bin_spectra(self, spectra, lmins, lmaxs):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)/2pi`.
        Return Cb
        """
        spectra = asarray(spectra)
        # minlmax = module_min(array([spectra.shape[-1] - 1, self.lmax]))
        minlmax = 30

        _p, _q = self._bin_operators(lmins, lmaxs)
        return dot(spectra[..., : minlmax + 1], _p.T[: minlmax + 1, ...])

    def bin_covariance(self, clcov, lmins, lmaxs):
        p, q = self._bin_operators(lmins, lmaxs, cov=True)
        return matmul(p, matmul(clcov, q))


def compute_offsets(ell, varcl, clref, fsky=1.0, iter=10):
    Nl = sqrt(abs(varcl - (2.0 / (2.0 * ell + 1) * clref**2) / fsky))
    for i in range(iter):
        Nl = sqrt(
            abs(varcl - 2.0 / (2.0 * ell + 1) / fsky * (clref**2 + 2.0 * Nl * clref))
        )
    return Nl * sqrt((2.0 * ell + 1) / 2.0)


def read_dl(datafile):
    data = np.loadtxt(datafile).T
    dl = array(zeros((3, 301)))  # EE,BB,EB #AGAIN HARDCODED FOR JAX COMPAT- LETS FIX THIS UP
    ell = array(data[0], int)

    if get_jax_enabled():
        dl = dl.at[0, ell].set(data[1])
        dl = dl.at[1, ell].set(data[2])
        dl = dl.at[2, ell].set(data[3])

    else:
        dl[0, ell] = data[1]
        dl[1, ell] = data[2]
        dl[2, ell] = data[3]
    return dl


def get_binning(lmin, lmax):
    dl = 10
    if lmin < 2:
        raise ValueError(f"Lmin should be > 2: {lmin}")
    if lmax > 200:
        raise ValueError(f"Lmax should be < 200: {lmax}")

    if lmin >= 36:
        lmins = list(range(lmin, lmax - dl + 2, dl))
        lmaxs = list(range(lmin + dl - 1, lmax + 1, dl))
    elif lmax <= 35:
        lmins = list(range(lmin, lmax + 1))
        lmaxs = list(range(lmin, lmax + 1))
    else:
        llmin = lmin
        llmax = 35
        hlmin = 36
        hlmax = lmax
        lmins = list(range(llmin, llmax + 1)) + list(range(hlmin, hlmax - dl + 2, dl))
        lmaxs = list(range(llmin, llmax + 1)) + list(
            range(hlmin + dl - 1, hlmax + 1, dl)
        )
    binc = Bins(lmins, lmaxs)
    return binc, lmins, lmaxs


def bin_covEB(clcov, binc, lmins, lmaxs):
    nell = len(clcov) // 3
    cbcov = array(zeros((3 * binc.nbins, 3 * binc.nbins)))
    for t1 in range(3):
        for t2 in range(3):
            mymat = array(zeros((binc.lmax + 1, binc.lmax + 1)))
            mymat[2:, 2:] = clcov[
                t1 * nell : t1 * nell + (binc.lmax - 1),
                t2 * nell : t2 * nell + (binc.lmax - 1),
            ]
            cbcov[
                t1 * binc.nbins : (t1 + 1) * binc.nbins,
                t2 * binc.nbins : (t2 + 1) * binc.nbins,
            ] = binc.bin_covariance(mymat, lmins, lmaxs)
    return cbcov


def bin_covBB(clcov, binc, lmins, lmaxs):
    nell = len(clcov) // 3
    t1 = t2 = 1
    mymat = array(zeros((binc.lmax + 1, binc.lmax + 1)))

    if get_jax_enabled():
        initial_arr = clcov[
            t1 * nell : t1 * nell + (binc.lmax - 1),
            t2 * nell : t2 * nell + (binc.lmax - 1),
        ]
        mymat = pad(
            initial_arr, pad_width=((2, 0), (2, 0)), mode="constant", constant_values=0
        )

    else:
        mymat[2:, 2:] = clcov[
            t1 * nell : t1 * nell + (binc.lmax - 1),
            t2 * nell : t2 * nell + (binc.lmax - 1),
        ]
    cbcov = binc.bin_covariance(mymat, lmins, lmaxs)
    return cbcov


def bin_covEE(clcov, binc, lmins, lmaxs):
    nell = len(clcov) // 3
    t1 = t2 = 0
    lmax = 30
    mymat = array(zeros((lmax + 1, lmax + 1)))

    if get_jax_enabled():
        initial_arr = clcov[
            t1 * nell : t1 * nell + (lmax - 1),
            t2 * nell : t2 * nell + (lmax - 1),
        ]
        mymat = pad(
            initial_arr, pad_width=((2, 0), (2, 0)), mode="constant", constant_values=0
        )

    else:
        mymat[2:, 2:] = clcov[
            t1 * nell : t1 * nell + (binc.lmax - 1),
            t2 * nell : t2 * nell + (binc.lmax - 1),
        ]

    cbcov = binc.bin_covariance(mymat, lmins, lmaxs)
    return cbcov


def vec2mat(vect):
    """
    shape EE, BB and EB as a matrix
    input:
        vect: EE,BB,EB
    output:
        matrix: [[EE,EB],[EB,BB]]
    """
    mat = array(zeros((2, 2)))
    if get_jax_enabled():
        mat = mat.at[0, 0].set(vect[0])
        mat = mat.at[1, 1].set(vect[1])
        if len(vect) == 3:
            mat = mat.at[1, 0].set(vect[2])
            mat = mat.at[0, 1].set(vect[2])

    else:
        mat[0, 0] = vect[0]
        mat[1, 1] = vect[1]
        if len(vect) == 3:
            mat[1, 0] = mat[0, 1] = vect[2]

    print("VEC2MAT MATRIX", mat)
    return mat


def mat2vec(mat):
    """
    shape polar matrix into polar vect
    input:
        matrix: [[EE,EB],[EB,BB]]
    output:
        vect: EE,BB,EB
    """
    vec = array([mat[0, 0], mat[1, 1], mat[0, 1]])
    return vec


def ghl(x):
    return sign(x - 1) * sqrt(2.0 * (x - log(x) - 1)) 