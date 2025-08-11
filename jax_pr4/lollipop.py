#
# LILLIPOP
#
# Sep 2020   - M. Tristram -

# Re factored by Alexander Reeves Oct 2023
from jax_pr4.module import *
import os
# # from UPanda.likelihoods import utils_likelihood
# from .likelihood import Likelihood

import astropy.io.fits as fits

# from UPanda.likelihoods import utils_likelihood
from jax_pr4 import utils
from jax_pr4 import tools_lollipop as tools

data_url = "https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods"


class LollipopPR4:
    # install_options = {"download_url": f"{data_url}/planck_2020_lollipop.tar.gz"}

    def __init__(
        self,
        data_vector=None,
        covariance_matrix=None,
        emu_model_list=None,
        dtype=None,
        compression_vecs=None,
        cosmology_params_fixed=[],
        additional_args={},
        external_spectra=None,
    ):
        self.additional_args = additional_args
        self.external_spectra = external_spectra

        # use fiducial lollipop set-up
        self.hartlap_factor = False
        self.marginalised_over_covariance = False
        self.Nsim = 400
        self.lmin = 2
        self.lmax = 30

        self.cl_file = "cl_lolEB_NPIPE.dat"
        self.fiducial_file = "fiducial_lolEB_planck2018_tensor_lensedCls.dat"
        self.cl_cov_file = "clcov_lolEB_NPIPE.fits"

        self.emu_model_list = emu_model_list
        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed

        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_directory)
        self.data_folder = os.path.join(parent_dir, "data/planck_pr4_lollipop")

        # Setting mode given likelihood name
        self.mode = "lowlE"
        if self.mode not in ["lowlE", "lowlB", "lowlEB"]:
            raise (
                self.log,
                "The '{} likelihood is not currently supported. Check your likelihood name.",
                self.mode,
            )

        # Binning (fixed binning)
        self.bins, lmins, lmaxs = tools.get_binning(self.lmin, self.lmax)

        self.lmins = lmins
        self.lmaxs = lmaxs
        # Data (ell,ee,bb,eb)
        filepath = os.path.join(self.data_folder, self.cl_file)
        data = tools.read_dl(filepath)
        # Files provide Cl for Lollipop; bin directly as Cl
        self.cldata = self.bins.bin_spectra(data, lmins, lmaxs)

        # Fiducial spectrum (ell,ee,bb,eb)
        print("Reading model")
        filepath = os.path.join(self.data_folder, self.fiducial_file)
        data = tools.read_dl(filepath)
        # Fiducial file provides Cl; bin directly as Cl
        self.clfid = self.bins.bin_spectra(data, lmins, lmaxs)

        # covmat (ee,bb,eb)
        print("Reading covariance")
        filepath = os.path.join(self.data_folder, self.cl_cov_file)
        clcov = fits.getdata(filepath)
        if self.mode == "lowlEB":
            cbcov = tools.bin_covEB(array(clcov), self.bins, lmins, lmaxs)
        elif self.mode == "lowlE":
            cbcov = tools.bin_covEE(array(clcov), self.bins, lmins, lmaxs)
        elif self.mode == "lowlB":
            cbcov = tools.bin_covBB(array(clcov), self.bins, lmins, lmaxs)
        clvar = diag(cbcov).reshape(-1, self.bins.nbins)

        if self.mode == "lowlEB":
            rcond = getattr(self, "rcond", 1e-9)
            self.invclcov = linalg.pinv(cbcov, rcond)
        else:
            import numpy as np
            self.invclcov = array(np.linalg.inv(cbcov))

        # Hartlap et al. 2008
        if self.hartlap_factor:
            if self.Nsim != 0:
                self.invclcov *= (self.Nsim - len(cbcov) - 2) / (self.Nsim - 1)

        if self.marginalised_over_covariance:
            if self.Nsim <= 1:
                raise (
                    self.log,
                    "Need the number of MC simulations used to compute the covariance in order to marginalise over (Nsim>1).",
                )

        # compute offsets
        print("Compute offsets")
        fsky = getattr(self, "fsky", 0.52)
        self.cloff = tools.compute_offsets(self.bins.lbin, clvar, self.clfid, fsky=fsky)

        if get_jax_enabled():
            self.cloff = self.cloff.at[2:].set(0.0)  # force NO offsets EB

        else:
            self.cloff[2:] = 0.0  # force NO offsets EB

        print("Initialized!")

    def _compute_chi2_2fields(self, cl, params_dict):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        # get model in Cl, muK^2
        # Bin input Cl spectra (batched: [batch, ell]) into [batch, nbins] per mode
        clth_stacked = []
        for mode in ["ee", "bb", "eb"]:
            if mode in cl:
                clth_stacked.append(self.bins.bin_spectra(cl[mode], self.lmins, self.lmaxs, input_is_dl=False))
        clth = array(clth_stacked)  # shape [nmode=3, batch, nbins]

        #if A_act is in params_dict, use it, otherwise use A_planck
        if "A_act" in params_dict:
            cal = array(params_dict["A_act"]) ** 2
        else:
            cal = array(params_dict["A_planck"]) ** 2

        # Prepare batched per-ell transformations
        nbins = self.cldata.shape[1]
        batch = cal.shape[0]
        x = array(zeros((batch, 3, nbins)))
        for ell in range(nbins):
            # Offsets are non-batched, shape (3,)
            Oo = tools.vec2mat(self.cloff[:, ell])  # 2x2

            # Data vector per batch
            d_vect = self.cldata[:, ell][newaxis, :] * cal[:, newaxis]  # [batch, 3]
            D = vmap(tools.vec2mat)(d_vect) + Oo  # [batch, 2, 2]

            m_vect = clth[:, :, ell].transpose(1, 0)  # [batch, 3]
            M = vmap(tools.vec2mat)(m_vect) + Oo  # [batch, 2, 2]

            f_vect = self.clfid[:, ell][newaxis, :].repeat(batch, axis=0)  # [batch, 3]
            F = vmap(tools.vec2mat)(f_vect) + Oo  # [batch, 2, 2]

            # compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
            w, V = linalg.eigh(M)
            #            if prod( sign(w)) <= 0:
            #                print( "WARNING: negative eigenvalue for l=%d" %l)
            L = V @ diag(1.0 / sqrt(w)) @ transpose(V, (0, 2, 1))
            P = transpose(L, (0, 2, 1)) @ D @ L

            # apply HL transformation
            w, V = linalg.eigh(P)
            g = sign(w) * tools.ghl(abs(w))
            G = V @ diag(g) @ transpose(V, (0, 2, 1))

            # cholesky fiducial
            w, V = linalg.eigh(F)
            L = V @ diag(sqrt(w)) @ transpose(V, (0, 2, 1))

            # compute C_fid^1/2 * G * C_fid^1/2
            X = transpose(L, (0, 2, 1)) @ G @ L
            x[:, :, ell] = vmap(tools.mat2vec)(X)

        # compute chi2
        # x = x.flatten()
        # if self.marginalised_over_covariance:
        #     chi2 = self.Nsim * log(1 + (x @ self.invclcov @ x) / (self.Nsim - 1))
        # else:
        #     chi2 = x @ self.invclcov @ x

        # update for 2d x inputs

        # Flatten per batch and compute quadratic form diag(X @ inv @ X^T)
        Xflat = x.reshape((batch, -1))
        if self.marginalised_over_covariance:
            quad = einsum("bi,ij,bj->b", Xflat, self.invclcov, Xflat)
            chi2 = self.Nsim * log(1 + quad / (self.Nsim - 1))
        else:
            chi2 = einsum("bi,ij,bj->b", Xflat, self.invclcov, Xflat)

        print(f"chi2/ndof = {chi2[0]}/{X.shape[1]}")

        return chi2

    def _compute_chi2_1field(self, cl, params_dict):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        # model in Cl, muK^2
        m = 0 if self.mode == "lowlE" else 1
        clth = self.bins.bin_spectra(
            cl["ee" if self.mode == "lowlE" else "bb"], self.lmins, self.lmaxs
        )

        #if A_act is in params_dict, use it, otherwise use A_planck
        if "A_act" in params_dict:
            cal = array(params_dict["A_act"]) ** 2
        else:
            cal = array(params_dict["A_planck"]) ** 2

        x = (self.cldata[m] * cal[:, newaxis] + self.cloff[m]) / (clth + self.cloff[m])
        g = sign(x) * tools.ghl(abs(x))

        X = (
            (sqrt(self.clfid[m] + self.cloff[m]))
            * g
            * (sqrt(self.clfid[m] + self.cloff[m]))
        )

        if self.marginalised_over_covariance:
            chi2 = self.Nsim * log(
                1 + sum(X @ self.invclcov * X, axis=1) / (self.Nsim - 1)
            )
        else:
            chi2 = matmul(self.invclcov, transpose(X))
            chi2 = matmul(X, chi2)
            chi2 = diag(chi2)
            print("XSHAPE", X.shape[1])

        # if self.marginalised_over_covariance:
        #     # marginalised over S = Ceff
        #     chi2 = self.Nsim * log(1 + (X @ self.invclcov @ X) / (self.Nsim - 1))
        # else:
        #     chi2 = X @ self.invclcov @ X

        # print(f"chi2/ndof = {chi2[0]}/{X.shape[1]}")

        # print("chi2 shape", chi2.shape)
        return chi2

    def get_requirements(self):
        return dict(Cl={mode: self.bins.lmax for mode in ["ee", "bb"]})

    def compute_like(self, ClEE, ClBB=None, ClEB=None, params=None):
        """
        Compute likelihood from input Cl spectra.
        
        This method computes the Lollipop low-ell likelihood,
        exactly matching the UPanda implementation.
        
        Args:
            ClEE: CMB EE spectrum (batch, ell_max+1)
            ClBB: CMB BB spectrum (batch, ell_max+1), optional
            ClEB: CMB EB spectrum (batch, ell_max+1), optional
            params: Dictionary of parameters
            
        Returns:
            Log-likelihood values
        """
        if params is None:
            params = {}
            
        # Convert to μK² units
        units_factor = 1e12
        
        # Add two extra zeros at the front for ell=0,1
        rows = ClEE.shape[0]
        zeros_column = zeros(rows)  # Single column of zeros
        
        Clee = column_stack((zeros_column, zeros_column, ClEE * units_factor))
        
        cl = {}
        cl["ee"] = Clee

        if ClBB is not None:
            cl["bb"] = column_stack((zeros_column, zeros_column, ClBB * units_factor))
        if ClEB is not None:
            cl["eb"] = column_stack((zeros_column, zeros_column, ClEB * units_factor))

        if self.mode == "lowlEB":
            chi2 = self._compute_chi2_2fields(cl, params)
        elif self.mode in ["lowlE", "lowlB"]:
            chi2 = self._compute_chi2_1field(cl, params)

        return -0.5 * chi2

    def set_external_spectra(self, spectra):
        self.external_spectra = spectra

    @classmethod
    def is_installed(cls, **kwargs):
        if kwargs.get("data", True):
            path = cls.get_path(kwargs["path"])
            if not (
                cls.get_install_options()
                and os.path.exists(path)
                and len(os.listdir(path)) > 0
            ):
                return False
            if not os.path.exists(os.path.join(path, "planck_2020/lollipop")):
                return False
        return True
