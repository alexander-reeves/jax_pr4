"""
Planck PR4/NPIPE CamSpec high-ℓ likelihood for JAX PR4

This is a JAX-compatible implementation of the Planck PR4 CamSpec likelihood
that takes direct Cl inputs rather than cosmological parameters.

The implementation is identical to UPanda's camspec_pr4.py except that
ClTT, ClTE, ClEE are passed directly instead of being computed from
cosmological parameters via emulators.
"""

from .module import *
from . import utils
import os

# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities (light copies from Cobaya)
# ─────────────────────────────────────────────────────────────────────────────

def _range_to_ells(use_range):
    if isinstance(use_range, str):
        rng = []
        for part in use_range.split():
            if '-' in part:
                lo, hi = map(int, part.split('-'))
                rng.append(range(lo, hi + 1))
            else:
                rng.append(int(part))
        return concatenate(rng)
    return asarray(use_range)

def _read_normalized(fname, pivot=None):
    import numpy as np
    dat = np.loadtxt(fname)
    assert int(dat[0, 0]) == 2
    dat = hstack(([0, 0], dat[:, 1]))
    if pivot is not None:
        dat /= dat[pivot]
    return dat

class CamSpecPR4:
    """
    Planck PR4/NPIPE CamSpec likelihood class.
    
    This class computes the likelihood for high-ℓ CMB power spectra
    using the CamSpec likelihood with optional low-ell components.
    
    The implementation is identical to UPanda's camspec_pr4.py except that
    ClTT, ClTE, ClEE are passed directly instead of being computed from
    cosmological parameters via emulators.
    """

    def __init__(self, additional_args=None):
        """
        Initialize CamSpec PR4 likelihood.
        
        Parameters:
        -----------
        additional_args : dict, optional
            Additional configuration arguments:
            - add_lowell: bool - Include low-ell TT and EE lognormal bins (default: False)
            - with_planck: bool - Include only low-ell TT (default: False)
        """
        if additional_args is None:
            additional_args = {}
            
        self.additional_args = additional_args
        
        # Configuration flags
        self.add_lowell = additional_args.get('add_lowell', False)
        self.with_planck = additional_args.get('with_planck', False)

        # Use relative path to data directory (go up one level from jax_pr4/ to data/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.data_dir = os.path.join(parent_dir, "data", "camspec_pr4")

        # Data files
        cl_hat_file        = os.path.join(self.data_dir, 'like_NPIPE_12.6_unified_spectra.txt')
        covmat_file        = os.path.join(self.data_dir, 'like_NPIPE_12.6_unified_cov.bin')
        data_ranges_file   = os.path.join(self.data_dir, 'like_NPIPE_12.6_unified_data_ranges.txt')

        # Load data and covariance
        self._load_data_and_covariance(cl_hat_file, covmat_file, data_ranges_file)
        
        # Load low-ell data if needed
        if self.add_lowell or self.with_planck:
            # Use relative path to low-ell data directory
            data_dir_lowl = os.path.join(parent_dir, "data", "planck_2018_lowl")
            self._load_low_ell_data(data_dir_lowl)
        
        print("CamSpec PR4 likelihood initialized successfully!")

    def _load_data_and_covariance(self, cl_hat_file, covmat_file, data_ranges_file):
        """Load CamSpec data and covariance matrix."""
        try:
            # Check if files exist
            if not all(os.path.exists(f) for f in [cl_hat_file, covmat_file, data_ranges_file]):
                print(f"⚠ CamSpec data files not found in {self.data_dir}")
                self.data_vector = None
                self.cov = None
                self.covinv = None
                return
            
            # Load spectra data
            import numpy as np
            spectra = np.loadtxt(cl_hat_file)
            
            # Load data ranges
            with open(data_ranges_file, 'r', encoding='utf-8-sig') as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            self.cl_names = []
            self.lmin = []
            self.lmax = []
            for ln in lines:
                tp, mn, mx = ln.split()
                self.cl_names.append(tp)
                self.lmin.append(int(mn))
                self.lmax.append(int(mx))
            self.lmin = asarray(self.lmin, int)
            self.lmax = asarray(self.lmax, int)

            # Calculate total size and used spectra
            use_cl_default = '143x143 217x217 143x217 TE EE'
            self.use_cl = use_cl_default.split()

            data_vec_all, used_indices = [], []
            nX = 0
            for tp, mn, mx in zip(self.cl_names, self.lmin, self.lmax):
                if mx < mn:
                    continue
                spec = spectra[mn:mx+1, self.cl_names.index(tp)]
                data_vec_all.append(spec)
                if tp in self.use_cl:
                    ells = arange(mn, mx+1)
                    used_indices.append(ells + (nX-mn))
                nX += mx - mn + 1

            # Set up ell ranges and used sizes
            self.ell_ranges = [None]*len(self.cl_names)
            self.used_sizes = [0]*len(self.cl_names)
            for idx, (tp, mn, mx) in enumerate(zip(self.cl_names, self.lmin, self.lmax)):
                if tp in self.use_cl:
                    ells = arange(mn, mx+1)
                    self.ell_ranges[idx] = ells
                    self.used_sizes[idx] = len(ells)

            # Create data vector
            self.data_vector = concatenate(data_vec_all)[concatenate(used_indices)]
            
            # Load covariance matrix (following UPanda approach)
            import numpy as np
            dtype_cov = np.float32
            cov_full = np.fromfile(covmat_file, dtype=dtype_cov).reshape(nX, nX).astype(np.float64)
            cov = cov_full[np.ix_(concatenate(used_indices), concatenate(used_indices))]
            self.cov = cov
            self.covinv = linalg.inv(self.cov)
            
            # Set up foreground templates (following UPanda approach)
            pivot = 1500
            max_l = module_max(self.lmax)
            self.ls = arange(max_l + 1)  # shape = (ℓmax+1,)
            self.llp1 = self.ls * (self.ls + 1)
            
            print(f"✓ CamSpec data loaded: {len(self.data_vector)} data points")
            print(f"  Covariance matrix shape: {self.cov.shape}")
            print(f"  Foreground array size: {self.ls.size}")
            
        except Exception as e:
            print(f"⚠ CamSpec data loading failed: {e}")
            self.data_vector = None
            self.cov = None
            self.covinv = None

    def _load_low_ell_data(self, data_dir_lowl):
        """Load low-ell lognormal data."""
        try:
            # Load EE data (3 bins)
            (
                self.lmin_list_EE,
                self.lmax_list_EE,
                self.mu_LN_EE,
                self.sig_LN_EE,
                self.loc_LN_EE,
            ) = utils.load_lognormal_data(data_dir_lowl, "lognormal_fit_3bins_EE.txt")
            
            # Load TT data (2 bins)
            (
                self.lmin_list_TT,
                self.lmax_list_TT,
                self.mu_LN_TT,
                self.sig_LN_TT,
            ) = utils.load_lognormal_data(data_dir_lowl, "lognormal_fit_2bins_TT.txt")
            
            print("✓ Low-ell data loaded successfully")
            
        except Exception as e:
            print(f"⚠ Low-ell data loading failed: {e}")

    def _get_cals(self, params):
        """Get calibration factors."""
        # 1. grab A_planck as a length-batch array
        default_vals = ones_like(params.get('A_planck', 1.0))
        A_planck = atleast_1d(params.get('A_planck', 1.0))
        A2 = A_planck**2  # shape = (batch,)

        # 2. for each cal-parameter, make it a length-batch array
        cal0 = atleast_1d(params.get('cal0', default_vals))
        cal2 = atleast_1d(params.get('cal2', default_vals))
        calTE = atleast_1d(params.get('calTE', default_vals))
        calEE = atleast_1d(params.get('calEE', default_vals))

        # 3. stack them into shape (batch,6)
        #    order: [cal0, 1, cal2, sqrt(cal2), calTE, calEE]
        cals = stack([
            cal0,
            ones_like(cal0),
            cal2,
            sqrt(cal2),
            calTE,
            calEE,
        ], axis=1)  # (batch,6)

        # 4. apply the A_planck² rescaling per sample
        return cals * A2[:, None]  # still (batch,6)

    def _get_foregrounds(self, params):
        """Get foreground contributions."""
        # Get batch size from A_planck parameter
        default_vals = ones_like(params.get('A_planck', 1.))
        
        # Extract foreground parameters
        amp_100 = params.get('amp_100', default_vals)
        amp_143 = params.get('amp_143', default_vals)
        amp_217 = params.get('amp_217', default_vals)
        amp_143x217 = params.get('amp_143x217', default_vals)
        
        n_100 = params.get('n_100', default_vals)
        n_143 = params.get('n_143', default_vals)
        n_217 = params.get('n_217', default_vals)
        n_143x217 = params.get('n_143x217', default_vals)
        
        # Stack them into (batch,4)
        amps = stack([amp_100, amp_143, amp_217, amp_143x217], axis=1)
        tilts = stack([n_100, n_143, n_217, n_143x217], axis=1)
        
        # Build cluster of ℓ by (ℓmax+1)
        ls = self.ls  # shape = (ℓmax+1,)
        pivot = 1500
        
        # For each of the 4 components, compute (batch,ℓmax+1)
        # result will be (batch,4,ℓmax+1)
        C = array(zeros((amps.shape[0], 4, ls.size)))
        print("IS JAX:", get_jax_enabled())
        for i in range(4):
            if get_jax_enabled(): 
                C = C.at[:, i, :].set(amps[:, i:i+1] * (ls[None, :]/pivot)**(tilts[:, i:i+1]))
            else:
                C[:, i, :] = amps[:, i:i+1] * (ls[None, :]/pivot)**(tilts[:, i:i+1])
        return C
        
    def get_prediction(self, ClTT, ClTE, ClEE, params=None):
        """
        Get theoretical prediction for CamSpec likelihood.
        
        Args:
            ClTT: CMB TT spectrum (batch, ell_max+1)
            ClTE: CMB TE spectrum (batch, ell_max+1) 
            ClEE: CMB EE spectrum (batch, ell_max+1)
            params: Dictionary of parameters
            
        Returns:
            Theoretical prediction vector
        """
        if params is None:
            params = {}
            
        # Convert to μK² units
        uK2 = 1e12
        ClTT = ClTT * uK2
        ClTE = ClTE * uK2
        ClEE = ClEE * uK2
        
        # Get batch size
        batch = ClTT.shape[0]
        
        # Store raw low-ℓ spectra (ℓ≥2 index 0→ℓ=2)
        self.Cltt = ClTT.copy()
        self.Clee = ClEE.copy()
        
        # Construct D_ell with zeros at ℓ=0,1
        ell_tt = arange(2, ClTT.shape[1] + 2)
        fac_tt = ell_tt * (ell_tt + 1) / (2 * pi)
        ell_te = arange(2, ClTE.shape[1] + 2)
        fac_te = ell_te * (ell_te + 1) / (2 * pi)
        ell_ee = arange(2, ClEE.shape[1] + 2)
        fac_ee = ell_ee * (ell_ee + 1) / (2 * pi)
        
        DTT = hstack((zeros((batch, 2)), ClTT * fac_tt))
        DTE = hstack((zeros((batch, 2)), ClTE * fac_te))
        DEE = hstack((zeros((batch, 2)), ClEE * fac_ee))
        
        # Get calibration and foregrounds
        cals = self._get_cals(params)
        fg = self._get_foregrounds(params) if hasattr(self, 'used_sizes') and any(self.used_sizes[:4]) else None
        

        pred = array(zeros((batch, self.data_vector.size)))
            
        ix = 0
        for i, (tp, n) in enumerate(zip(self.cl_names, self.used_sizes)):
            if n == 0:
                continue
            ells = self.ell_ranges[i]
            spec = (DTT if i <= 3 else DTE if i == 4 else DEE)[:, ells]  # (batch, n)
            if i <= 3 and fg is not None:
                spec += fg[:, i, ells]
            spec /= cals[:, i:i+1]
            if get_jax_enabled():
                pred = pred.at[:, ix:ix+n].set(spec)
            else:
                pred[:, ix:ix+n] = spec
            ix += n
            
        return pred

    def compute_like(self, ClTT, ClTE, ClEE, params=None):
        """
        Compute log-likelihood from input Cl spectra.
        
        This method computes the CamSpec high-ell likelihood plus optional
        low-ell lognormal components, exactly matching the UPanda implementation.
        """
        if params is None:
            params = {}
        
        # Get theoretical prediction
        pred = self.get_prediction(ClTT, ClTE, ClEE, params)
        
        # High-ell CamSpec likelihood
        diff = pred - self.data_vector
        
        # Check for binning/compression
        binning_matrix = self.additional_args.get("binning_matrix", None)
        binned_covinv = self.additional_args.get("binned_covinv", None)
        
        if binning_matrix is not None:
            # Bin the residuals directly (consistent with hillipop approach)
            diff_binned = binning_matrix @ diff.T  # (n_bins, batch)
            if binned_covinv is not None:
                covinv_binned = binned_covinv
            else:
                # Correctly bin the covariance matrix first, then invert
                cov_binned = binning_matrix @ self.cov @ binning_matrix.T
                covinv_binned = linalg.inv(cov_binned)
            # Compute chi2 in binned space
            like = -0.5 * (einsum('bi,ij,jb->b', diff_binned.T, covinv_binned, diff_binned))
        else:
            # Standard chi-squared calculation
            like = -0.5 * (einsum('bi,ij,bj->b', diff, self.covinv, diff))
        
        # Low-ell lognormal contribution
        if self.add_lowell or self.with_planck:
            # Rebuild params dict for low-ell calculation
            A2 = (array(params['A_planck'])[:, None]**2) if like.ndim > 0 else params['A_planck']**2
            Cls_tt_2_29 = (self.Cltt / A2)[:, 0:28]  # ell=2-29
            Cls_ee_2_29 = (self.Clee / A2)[:, 0:28]  # ell=2-29
            
            if self.add_lowell:
                print("lowell")
                ln_like_ee = utils.planck_lowE_binned_loglike(
                    Cls_ee_2_29, self.mu_LN_EE, self.sig_LN_EE, self.loc_LN_EE,
                    self.lmin_list_EE, self.lmax_list_EE)
                ln_like_tt = utils.planck_lowT_binned_loglike(
                    Cls_tt_2_29, self.mu_LN_TT, self.sig_LN_TT,
                    self.lmin_list_TT, self.lmax_list_TT)
                like += ln_like_tt + ln_like_ee
            elif self.with_planck:
                print("with planck")
                ln_like_tt = utils.planck_lowT_binned_loglike(
                    Cls_tt_2_29, self.mu_LN_TT, self.sig_LN_TT,
                    self.lmin_list_TT, self.lmax_list_TT)
                like += ln_like_tt
        
        return like 