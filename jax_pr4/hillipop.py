#
# HILLIPOP
#
# Sep 2020   - M. Tristram -

# Re factored by Alexander Reeves Oct 2023
from jax_pr4.module import *
import os
from itertools import combinations
import astropy.io.fits as fits
import numpy as np
# Import from jax_pr4 modules
from jax_pr4 import utils
from jax_pr4 import foregrounds_hillipop as fg


# list of available foreground models
fg_list = {
    "sbpx": fg.subpix,
    "ps": fg.ps,
    "dust": fg.dust,
    "dust_model": fg.dust_model,
    "sync": fg.sync_model,
    "ksz": fg.ksz_model,
    "ps_radio": fg.ps_radio,
    "ps_dusty": fg.ps_dusty,
    "cib": fg.cib_model,
    "tsz": fg.tsz_model,
    "szxcib": fg.szxcib_model,
}

# def set_provider(self, provider):
#     """
#     Set the provider for this likelihood.
#     """
#     self.provider = provider
# ------------------------------------------------------------------------------------------------
# Likelihood
# ------------------------------------------------------------------------------------------------

data_url = "https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods"


def load_data_jax(current_directory, filename):
    data_dir_lowl = os.path.join(current_directory, "data/planck_2018_lowl")
    file_path = os.path.join(data_dir_lowl, filename)

    data = []
    with open(file_path, "r") as file:
        next(file)  # skip header line
        for line in file:
            line_data = [float(item) for item in line.split()]
            data.append(line_data)

    # Transpose data to unpack correctly
    data_transposed = list(zip(*data))

    # Convert to JAX arrays
    lmin_list_TT, lmax_list_TT, mu_LN_TT, sig_LN_TT = [
        array(x) for x in data_transposed
    ]

    return lmin_list_TT, lmax_list_TT, mu_LN_TT, sig_LN_TT


class HillipopPR4:
    # multipoles_range_file: Optional[str]
    # xspectra_basename: Optional[str]
    # covariance_matrix_file: Optional[str]
    # foregrounds: Optional[list]

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
        self.emu_model_list = emu_model_list
        self.dtype = dtype
        self.cosmology_params_fixed = cosmology_params_fixed
        self.additional_args = additional_args  # in case you want to add the lowl tt lognorm bins from Planck 2018
        self.external_spectra = external_spectra

        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_directory)
        self.data_folder = os.path.join(parent_dir, "data/planck_pr4_hillipop")

        self.multipoles_range_file = "binning_v4.2.fits"
        self.xspectra_basename = "dl_PR4_v4.2"
        self.covariance_matrix_file = "invfll_PR4_v4.2_TTTEEE.fits"

        # do we want to set some ellmax?
        self.user_lmax = self.additional_args.get("lmax", {})

        # lognorm bins from planck2018 TT
        if get_jax_enabled():
            (
                self.lmin_list_TT,
                self.lmax_list_TT,
                self.mu_LN_TT,
                self.sig_LN_TT,
            ) = load_data_jax(parent_dir, "lognormal_fit_2bins_TT.txt")

        else:
            import numpy as np
            data_dir_lowl = os.path.join(parent_dir, "data/planck_2018_lowl")
            (
                lmin_list_TT,
                lmax_list_TT,
                mu_LN_TT,
                sig_LN_TT,
            ) = np.loadtxt(
                data_dir_lowl + "/lognormal_fit_" + "2" + "bins_TT.txt", unpack=True
            )
            # Convert to JAX arrays if needed
            self.lmin_list_TT = array(lmin_list_TT)
            self.lmax_list_TT = array(lmax_list_TT)
            self.mu_LN_TT = array(mu_LN_TT)
            self.sig_LN_TT = array(sig_LN_TT)

        # default set up from hillipop
        self.foregrounds = {
            "TT": {
                "dust_model": "foregrounds/DUST_Planck_PR4_model_v4.2",
                "tsz": "foregrounds/SZ_Planck_PR4_model.txt",
                "ksz": "foregrounds/kSZ_Planck_PR4_model.txt",
                "cib": "foregrounds/CIB_Planck_PR4_model.txt",
                "szxcib": "foregrounds/SZxCIB_Planck_PR4_model.txt",
                "ps_radio": None,
                "ps_dusty": None,
            },
            "EE": {"dust_model": "foregrounds/DUST_Planck_PR4_model_v4.2"},
            "TE": {"dust_model": "foregrounds/DUST_Planck_PR4_model_v4.2"},
        }

        self.frequencies = [100, 100, 143, 143, 217, 217]
        self._mapnames = ["100A", "100B", "143A", "143B", "217A", "217B"]
        self._nmap = len(self.frequencies)
        self._nfreq = len(set(self.frequencies))
        self._nxfreq = self._nfreq * (self._nfreq + 1) // 2
        self._nxspec = self._nmap * (self._nmap - 1) // 2
        self._xspec2xfreq = self._xspec2xfreq()

        # Get likelihood name and add the associated mode
        # likelihood_name = self.__class__.__name__
        # likelihood_modes = [
        #     likelihood_name[i : i + 2] for i in range(0, len(likelihood_name), 2)
        # ]

        likelihood_modes = ["TT", "TE", "EE"]  # fixed TTTEEE for now

        self._is_mode = {
            mode: mode in likelihood_modes for mode in ["TT", "TE", "EE", "BB"]
        }
        self._is_mode["ET"] = self._is_mode["TE"]

        # Multipole ranges
        filename = os.path.join(self.data_folder, self.multipoles_range_file)
        self._lmins, self._lmaxs = self._set_multipole_ranges(filename)
        lmax = module_max(array([max(ell) for ell in self._lmaxs.values()]))

        self.lmax = lmax

        # Data
        basename = os.path.join(self.data_folder, self.xspectra_basename)
        self._dldata = self._read_dl_xspectra(basename)

        # Weights
        dlsig = self._read_dl_xspectra(basename, hdu=2)

        for m, w8 in dlsig.items():
            if get_jax_enabled():
                w8 = where(w8 == 0, inf, w8)
            else:
                w8[w8 == 0] = inf

        self._dlweight = {k: 1 / v**2 for k, v in dlsig.items()}
        #        self._dlweight = ones(shape(self._dldata))

        # Inverted Covariance matrix
        filename = os.path.join(self.data_folder, self.covariance_matrix_file)
        # Sanity check
        # m = re.search(".*_(.+?).fits", self.covariance_matrix_file)

        # Only load the full inverse covariance matrix if not using compressed version
        binning_matrix = getattr(self, "additional_args", {}).get("binning_matrix", None)
        binned_invkll = getattr(self, "additional_args", {}).get("binned_invkll", None)
        compression = getattr(self, "additional_args", {}).get("hillipop_compression", None)

        if (binning_matrix is not None and binned_invkll is not None) or (binning_matrix is not None) or (compression is not None):
            self._invkll = None  # Not needed, will use compressed version
        else:
            filename = os.path.join(self.data_folder, self.covariance_matrix_file)
            self._invkll = self._read_invcovmatrix(filename).astype(np.float32)

        # # now lets set some ell max if desired
        # kill = []                   # indices to mask in data-vector & covmat
        # cursor = 0                  # current position in the flattened vector

        # for mode in ["TT", "EE", "TE"]:            # ET shares the TE limits
        #     if not self._is_mode[mode]:
        #         print("continue - mode not used", mode)
        #         continue

        #     for xf in range(self._nxfreq):         # six cross-frequency spectra
        #         lmin = self._lmins[mode][xf]
        #         lmax_native = self._lmaxs[mode][xf]        # 2500, 2000, …
        #         lmax_user   = self.user_lmax.get(mode, lmax_native)

        #         n_keep = max(0, min(lmax_user, lmax_native) - lmin + 1)
        #         n_tot  = lmax_native - lmin + 1

        #         # any band-power above the user's ℓ_max is discarded
        #         if n_keep < n_tot:
        #             print("removing ells - are you sure you want this?")
        #             kill.extend(range(cursor + n_keep, cursor + n_tot))

        #         cursor += n_tot
        # # --- inflate variance for discarded multipoles -----------------
        # # need to do this on the covmat...
        # if kill:
        #         print("re-making cut covariance. This may take a couple of minutes...")
        #         # self._invkll[kill, :] = 0.0
        #         # self._invkll[:, kill] = 0.0  
        #         cov  = np.linalg.inv(self._invkll)
        #         huge = 1e10
        #         cov[kill, kill] += huge
        #         self._invkll = np.linalg.inv(cov).astype("float32")
        #         print("done!")

        # Foregrounds
        self.fgs = {}  # list of foregrounds per mode [TT,EE,TE,ET]
        # Init foregrounds TT
        fgsTT = []
        if self._is_mode["TT"]:
            for name in self.foregrounds["TT"].keys():
                print(f"Adding '{name}' foreground for TT")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies, mode="TT")
                if isinstance(self.foregrounds["TT"][name], str):
                    kwargs["filename"] = os.path.join(
                        self.data_folder, self.foregrounds["TT"][name]
                    )
                elif name == "szxcib":
                    filename_tsz = self.foregrounds["TT"]["tsz"] and os.path.join(
                        self.data_folder, self.foregrounds["TT"]["tsz"]
                    )
                    filename_cib = self.foregrounds["TT"]["cib"] and os.path.join(
                        self.data_folder, self.foregrounds["TT"]["cib"]
                    )
                    kwargs["filenames"] = (filename_tsz, filename_cib)
                fgsTT.append(fg_list[name](**kwargs))
        self.fgs["TT"] = fgsTT

        # Init foregrounds EE
        fgsEE = []
        if self._is_mode["EE"]:
            for name in self.foregrounds["EE"].keys():
                print(f"Adding '{name}' foreground for EE")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies)
                if isinstance(self.foregrounds["EE"][name], str):
                    kwargs["filename"] = os.path.join(
                        self.data_folder, self.foregrounds["EE"][name]
                    )
                fgsEE.append(fg_list[name](mode="EE", **kwargs))
        self.fgs["EE"] = fgsEE

        # Init foregrounds TE
        fgsTE = []
        fgsET = []
        if self._is_mode["TE"]:
            for name in self.foregrounds["TE"].keys():
                print(f"Adding '{name}' foreground for TE")
                kwargs = dict(lmax=lmax, freqs=self.frequencies)
                if isinstance(self.foregrounds["TE"][name], str):
                    kwargs["filename"] = os.path.join(
                        self.data_folder, self.foregrounds["TE"][name]
                    )
                fgsTE.append(fg_list[name](mode="TE", **kwargs))
                fgsET.append(fg_list[name](mode="ET", **kwargs))
        self.fgs["TE"] = fgsTE
        self.fgs["ET"] = fgsET

        print("Initialized!")

    def _xspec2xfreq(self):
        list_fqs = []
        for f1 in range(self._nfreq):
            for f2 in range(f1, self._nfreq):
                list_fqs.append((f1, f2))

        if get_jax_enabled():
            freqs = [100, 143, 217]
        else:
            freqs = list(unique(self.frequencies))
        spec2freq = []
        for m1 in range(self._nmap):
            for m2 in range(m1 + 1, self._nmap):
                f1 = freqs.index(self.frequencies[m1])
                f2 = freqs.index(self.frequencies[m2])
                spec2freq.append(list_fqs.index((f1, f2)))

        return spec2freq

    def _set_multipole_ranges(self, filename):
        """
        Return the (lmin,lmax) for each cross-spectra for each mode (TT, EE, TE, ET)
        array(nmode,nxspec)
        """
        print("Define multipole ranges")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        # tags = ["TT", "EE", "BB", "TE"]
        lmins = {}
        lmaxs = {}
        with fits.open(filename) as hdus:
            for hdu in hdus[1:]:
                tag = hdu.header["spec"]
                lmins[tag] = hdu.data.LMIN
                lmaxs[tag] = hdu.data.LMAX
                # if self._is_mode[tag]:
                # print(f"{tag}")
                # print(f"lmin: {lmins[tag]}")
                # print(f"lmax: {lmaxs[tag]}")
        lmins["ET"] = lmins["TE"]
        lmaxs["ET"] = lmaxs["TE"]

        return lmins, lmaxs

    def _read_dl_xspectra(
        self, basename, hdu=1
    ):  # AR NEED TO UPDATE THIS FUNCTION TO MAKE LESS HARDCODED
        """
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl (TT,EE,TE,ET) in muK^2
        """
        print("Reading cross-spectra")

        lmax = 2500

        num_combinations = len(list(combinations(self._mapnames, 2)))
        dldata = array(zeros(
            (num_combinations, 4, lmax + 1)
        ))  # hardcoded for JAX compatibility- change this!

        with fits.open(
            f"{basename}_{self._mapnames[0]}x{self._mapnames[1]}.fits"
        ) as hdus:
            nhdu = len(hdus)
        if nhdu < hdu:
            # no sig in file, uniform weight
            print("Warning: uniform weighting for combining spectra !")
            dldata = ones((self._nxspec, 4, self.lmax + 1))
        else:
            if nhdu == 1:
                hdu = 0  # compatibility

            # dldata = []
            # for m1, m2 in combinations(self._mapnames, 2):
            #     data = fits.getdata(f"{basename}_{m1}x{m2}.fits", hdu) * 1e12
            #     print("DATA", data)
            #     tmpcl = list(data[[0, 1, 3], : self.lmax + 1])
            #     data = fits.getdata(f"{basename}_{m2}x{m1}.fits", hdu) * 1e12
            #     tmpcl.append(data[3, : self.lmax + 1])
            #     dldata.append(tmpcl)
            for idx, (m1, m2) in enumerate(combinations(self._mapnames, 2)):
                data1 = fits.getdata(f"{basename}_{m1}x{m2}.fits", hdu) * 1e12
                data2 = fits.getdata(f"{basename}_{m2}x{m1}.fits", hdu) * 1e12

                if get_jax_enabled():
                    dldata = dldata.at[idx, :3, :].set(data1[[0, 1, 3], : lmax + 1])
                    dldata = dldata.at[idx, 3, :].set(data2[3, : lmax + 1])
                else:
                    dldata[idx, :3, :] = data1[[0, 1, 3], : self.lmax + 1]
                    dldata[idx, 3, :] = data2[3, : self.lmax + 1]

        dldata = transpose(array(dldata), (1, 0, 2))
        return dict(zip(["TT", "EE", "TE", "ET"], dldata))

    def _read_invcovmatrix(self, filename, nel=29758):
        """
        Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
        Output: invkll [Dl in muK^-4]
        """
        print(f"Covariance matrix file: {filename}")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        data = fits.getdata(filename)
        data = data.reshape((nel, nel)) / 1e24  # muK^-4

        # comment for Jax compat for now
        # nell = self._get_matrix_size()
        # if nel != nell:
        #     raise ValueError(
        #         f"Incoherent covariance matrix (read:{nel}, expected:{nell})"
        #     )

        return data

    def _get_matrix_size(self):
        """
        Compute covariance matrix size given activated mode
        Return: number of multipole
        """
        nell = 0

        # TT,EE,TEET
        for m in ["TT", "EE", "TE"]:
            if self._is_mode[m]:
                nells = self._lmaxs[m] - self._lmins[m] + 1
                nell += sum(
                    array(
                        [nells[self._xspec2xfreq.index(k)] for k in range(self._nxfreq)]
                    )
                )

        return nell

    def _select_spectra(self, cl, mode):
        """
        Cut spectra given Multipole Ranges and flatten
        Return: 2D array
        """
        acl = asarray(cl)
        for xf in range(self._nxfreq):
            lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
            lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]

            if xf == 0:
                xl = acl[xf, :, lmin : lmax + 1]

            else:
                xl = hstack((xl, acl[xf, :, lmin : lmax + 1]))

        return xl


    def _xspectra_to_xfreq(self, cl, weight, normed=True):
        """
        Average cross-spectra per cross-frequency.

        If weight == inf (masked multipole) we just drop that multipole
        from both numerator and denominator instead of giving it weight 1.
        """
        xcl = array(zeros((self._nxfreq, cl.shape[1], 2500 + 1)))
        xw8 = array(zeros((self._nxfreq, cl.shape[1], 2500 + 1)))

        for xs in range(self._nxspec):
            w = weight[xs]
            if get_jax_enabled():
                finite = isfinite(w)
                w_safe = where(finite, w, 0.0)        # 0 where inf/NaN
                xcl = xcl.at[self._xspec2xfreq[xs]].add(w_safe * cl[xs])
                xw8 = xw8.at[self._xspec2xfreq[xs]].add(w_safe)
            else:
                finite = np.isfinite(w)
                w_safe = np.where(finite, w, 0.0)
                xcl[self._xspec2xfreq[xs]] += w_safe * cl[xs]
                xw8[self._xspec2xfreq[xs]] += w_safe

        if not normed:
            return xcl, xw8

        if get_jax_enabled():
            return where(xw8 > 0, xcl / xw8, 0.0)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(xw8 > 0, xcl / xw8, 0.0)
            return out

    # def _xspectra_to_xfreq(self, cl, weight, normed=True):
    #     """
    #     Average cross-spectra per cross-frequency
    #     """
    #     xcl = array(zeros((self._nxfreq, cl.shape[1], 2500 + 1)))
    #     xw8 = array(zeros((self._nxfreq, cl.shape[1], 2500 + 1)))

    #     #if weights are infs replace with 1s 
    #     weight = where(isinf(weight), 1, weight)
    #     for xs in range(self._nxspec):
    #         if get_jax_enabled():
    #             xcl = xcl.at[self._xspec2xfreq[xs]].add(weight[xs] * cl[xs])
    #             xw8 = xw8.at[self._xspec2xfreq[xs]].add(weight[xs])

    #         else:
    #             xcl[self._xspec2xfreq[xs]] += weight[xs] * cl[xs]
    #             xw8[self._xspec2xfreq[xs]] += weight[xs]

    #     if get_jax_enabled():             
    #         non_zero_weight_indices = xw8 != 0
    #         safe_xw8 = where(non_zero_weight_indices, xw8, 1)  # Replace 0s with 1s for safety
    #         inf_indices = xw8 == inf
    #         safe_xw8 = where(inf_indices, 1, xw8)  # Replace infs with 1s for safety
    #     else:
    #         xw8[xw8 == 0] = inf
    #     if normed:
    #         if get_jax_enabled(): 
    #             # normalized_xcl = where(non_zero_weight_indices, xcl / safe_xw8, 0)
    #             normalized_xcl = xcl/safe_xw8 #zero weight points should alreadyb be zero 
    #             return normalized_xcl
    #         else:
    #             return xcl / xw8
    #     else:
    #         return xcl, xw8
        
    

    def _get_delta_cl(self, pars):

        dlth, params_dict = self._compute_theory_cls(pars)

        Rspec = self._compute_residuals(params_dict, dlth, "TT")
        # average to cross-spectra
        Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["TT"], normed=True)
        # select multipole range
        Xl = self._select_spectra(Rl, "TT")

        if self._is_mode["EE"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_dict, dlth, "EE")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["EE"], normed=True)
            # select multipole range
            Xl = hstack((Xl, self._select_spectra(Rl, "EE")))

        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute residuals Rl = Dl - Dlth
            if self._is_mode["TE"]:
                Rspec = self._compute_residuals(params_dict, dlth, "TE")
                RlTE, WlTE = self._xspectra_to_xfreq(
                    Rspec, self._dlweight["TE"], normed=False
                )
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                Rspec = self._compute_residuals(params_dict, dlth, "ET")
                RlET, WlET = self._xspectra_to_xfreq(
                    Rspec, self._dlweight["ET"], normed=False
                )
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range

            #quickly make Rl and Wl "safe" for Jax 
            Wl = where(Wl !=0, Wl, 1)
            Wl = where(Wl != inf, Wl, 1)
            Xl = hstack((Xl, self._select_spectra(Rl / Wl, "TE")))

        self.delta_cl = asarray(Xl)

        return self.delta_cl


    def _compute_residuals(self, pars, dlth, mode):
        # Nuisances
        cal = []
        for m1, m2 in combinations(self._mapnames, 2):
            if mode == "TT":
                cal1, cal2 = array(pars[f"cal{m1}"]), array(pars[f"cal{m2}"])
                # print("cal TT", cal1, cal2)
            elif mode == "EE":
                cal1, cal2 = (
                    array(pars[f"cal{m1}"]) * array(pars[f"pe{m1}"]),
                    array(pars[f"cal{m2}"]) * array(pars[f"pe{m2}"]),
                )
            elif mode == "TE":
                cal1, cal2 = array(pars[f"cal{m1}"]), array(
                    pars[f"cal{m2}"] * pars[f"pe{m2}"]
                )
            elif mode == "ET":
                cal1, cal2 = array(pars[f"cal{m1}"] * pars[f"pe{m1}"]), array(
                    pars[f"cal{m2}"]
                )
            cal.append(cal1 * cal2 / array(pars["A_planck"]) ** 2)

        # Data
        dldata = self._dldata[mode]

        # # Model
        # dlmodel = [dlth[mode]] * self._nxspec

        # for fg_ in self.fgs[mode]:
        #     dlmodel_component = fg_.compute_dl(pars)
        #     dlmodel = array(dlmodel) + array(dlmodel_component)
        dlmodel = [dlth[mode]] * self._nxspec
        for fg_ in self.fgs[mode]:
            dlmodel_component = fg_.compute_dl(pars)
            # # Convert to list for addition like official version
            # if isinstance(dlmodel_component, array):
            dlmodel_component = dlmodel_component.tolist()
            dlmodel = [d1 + d2 for d1, d2 in zip(dlmodel, dlmodel_component)]
        # Convert back to array at the end
        dlmodel = array(dlmodel)

        Rspec = array(
            [
                dldata[xs] - cal[xs][:, newaxis] * dlmodel[xs]
                for xs in range(self._nxspec)
            ]
        )
        return Rspec

    def dof(self):
        return len(self._invkll)

    def reduction_matrix(self, mode):
        """
        Reduction matrix

        each column is equal to 1 in the 15 elements corresponding to a cross-power spectrum
        measurement in that multipole and zero elsewhere

        """
        X = array(zeros((len(self.delta_cl), self.lmax + 1)))
        x0 = 0
        for xf in range(self._nxfreq):
            lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
            lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]
            for il, l in enumerate(range(lmin, lmax + 1)):
                X[x0 + il, l] = 1
            x0 += lmax - lmin + 1

        return X

    def compute_chi2(self, dlth, params_dict):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        params_dict: Disctionary of parameter values

        dl: array or arr2d
              CMB power spectrum (Dl in muK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """

        # cl_boltz from Boltzmann (Cl in muK^2)
        #        lth = arange(self.lmax + 1)
        #        dlth = asarray(dl)[:, lth][[0, 1, 3, 3]]  # select TT,EE,TE,TE

        Rspec = self._compute_residuals(params_dict, dlth, "TT")
        # average to cross-spectra
        Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["TT"], normed=True)
        # select multipole range
        Xl = self._select_spectra(Rl, "TT")

        if self._is_mode["EE"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_dict, dlth, "EE")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["EE"], normed=True)
            # select multipole range
            Xl = hstack((Xl, self._select_spectra(Rl, "EE")))

        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute residuals Rl = Dl - Dlth
            if self._is_mode["TE"]:
                Rspec = self._compute_residuals(params_dict, dlth, "TE")
                RlTE, WlTE = self._xspectra_to_xfreq(
                    Rspec, self._dlweight["TE"], normed=False
                )
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                Rspec = self._compute_residuals(params_dict, dlth, "ET")
                RlET, WlET = self._xspectra_to_xfreq(
                    Rspec, self._dlweight["ET"], normed=False
                )
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range

            #quickly make Rl and Wl "safe" for Jax 
            # Wl = where(Wl !=0, Wl, 1)
            # Wl = where(Wl != inf, Wl, 1)
            Wl = where(Wl == 0, inf, Wl)
            Xl = hstack((Xl, self._select_spectra(Rl / Wl, "TE")))

        self.delta_cl = asarray(Xl).astype('float32')

        # Add mode masking based on additional_args
        if self.additional_args.get("tt_only"):
            # Keep only TT part, zero out the rest
            # Get TT size by computing TT residuals and selecting spectra
            Rspec_tt = self._compute_residuals(params_dict, dlth, "TT")
            Rl_tt = self._xspectra_to_xfreq(Rspec_tt, self._dlweight["TT"], normed=True)
            tt_spectra = self._select_spectra(Rl_tt, "TT")
            tt_size = array(tt_spectra).size  # Use .size to get total number of elements
            # Zero out everything after TT part for all batches
            self.delta_cl[:, tt_size:] = 0.0

        elif self.additional_args.get("ee_only"):
            # Keep only EE part, zero out the rest
            # Get TT size
            Rspec_tt = self._compute_residuals(params_dict, dlth, "TT")
            Rl_tt = self._xspectra_to_xfreq(Rspec_tt, self._dlweight["TT"], normed=True)
            tt_spectra = self._select_spectra(Rl_tt, "TT")
            tt_size = array(tt_spectra).size
            # Get EE size
            Rspec_ee = self._compute_residuals(params_dict, dlth, "EE")
            Rl_ee = self._xspectra_to_xfreq(Rspec_ee, self._dlweight["EE"], normed=True)
            ee_spectra = self._select_spectra(Rl_ee, "EE")
            ee_size = array(ee_spectra).size
            # Zero out TT part and everything after EE part for all batches
            self.delta_cl[:, :tt_size] = 0.0
            self.delta_cl[:, tt_size+ee_size:] = 0.0

        elif self.additional_args.get("te_only"):
            # Keep only TE part, zero out the rest
            # Get TT size
            Rspec_tt = self._compute_residuals(params_dict, dlth, "TT")
            Rl_tt = self._xspectra_to_xfreq(Rspec_tt, self._dlweight["TT"], normed=True)
            tt_spectra = self._select_spectra(Rl_tt, "TT")
            tt_size = array(tt_spectra).size
            # Get EE size
            Rspec_ee = self._compute_residuals(params_dict, dlth, "EE")
            Rl_ee = self._xspectra_to_xfreq(Rspec_ee, self._dlweight["EE"], normed=True)
            ee_spectra = self._select_spectra(Rl_ee, "EE")
            ee_size = array(ee_spectra).size
            # Zero out TT and EE parts for all batches
            self.delta_cl[:, :tt_size+ee_size] = 0.0

        # Check for compression
        # If both binning_matrix and binned_invkll are provided, use pre-computed binned covariance
        # Otherwise, use the original compression methods
        binning_matrix = getattr(self, "additional_args", {}).get("binning_matrix", None)
        binned_invkll = getattr(self, "additional_args", {}).get("binned_invkll", None)
        compression = getattr(self, "additional_args", {}).get("hillipop_compression", None)
        
        if binning_matrix is not None and binned_invkll is not None:
            # Use pre-computed binned inverse covariance matrix
            proj = binning_matrix @ self.delta_cl.T
            chi2 = (proj.T @ binned_invkll @ proj).diagonal()
            return chi2
        elif binning_matrix is not None:
            # Use binning matrix for compression (original method)
            projection_matrix = binning_matrix
            proj = projection_matrix @ self.delta_cl.T
            invkll_proj = projection_matrix @ self._invkll @ projection_matrix.T
            intermediate = invkll_proj @ proj
            chi2 = (proj.T @ intermediate).diagonal()
            return chi2
        elif compression is not None:
            # Use PCA or other compression
            projection_matrix = compression
            proj = projection_matrix @ self.delta_cl.T
            invkll_proj = projection_matrix @ self._invkll @ projection_matrix.T
            intermediate = invkll_proj @ proj
            chi2 = (proj.T @ intermediate).diagonal()
            return chi2

        def compute_chi2_jax(delta_cl, invkll):
            # # Matrix multiplication
            # intermediate = dot(invkll, delta_cl.T)

            # # Computing the diagonal elements efficiently
            # # Calculate chi2 using einsum
            # chi2 = einsum("ij,ji->i", delta_cl, intermediate)

            chi2 = einsum("bi,ij,bj->b", delta_cl, invkll, delta_cl)

            return chi2

        chi2 = compute_chi2_jax(self.delta_cl, self._invkll)

        # chi2 = self._invkll.dot(self.delta_cl[0]).dot(self.delta_cl[0])
        return chi2

    def get_requirements(self):
        return dict(Cl={mode: self.lmax for mode in ["tt", "ee", "te"]})

    def compute_like(self, ClTT, ClTE, ClEE, params=None):
        """
        Compute likelihood from input Cl spectra.
        
        This method computes the Hillipop likelihood plus optional
        low-ell lognormal components, exactly matching the UPanda implementation.
        
        Args:
            ClTT: CMB TT spectrum (batch, ell_max+1)
            ClTE: CMB TE spectrum (batch, ell_max+1) 
            ClEE: CMB EE spectrum (batch, ell_max+1)
            params: Dictionary of parameters
            
        Returns:
            Log-likelihood values
        """
        if params is None:
            params = {}
            
        # Convert to μK² units
        units_factor = 1e12
        
        # Add two extra zeros at the front for ell=0,1
        rows = ClTT.shape[0]
        zeros_column = zeros(rows)  # Single column of zeros

        Cltt = column_stack((zeros_column, zeros_column, ClTT * units_factor))
        Clte = column_stack((zeros_column, zeros_column, ClTE * units_factor))
        Clee = column_stack((zeros_column, zeros_column, ClEE * units_factor))

        lmax = 2500
        ell = arange(lmax + 1)

        dlth = {}
        dlth["TT"] = Cltt[:, : lmax + 1] * (ell * (ell + 1)) / (2 * pi)
        dlth["EE"] = Clee[:, : lmax + 1] * (ell * (ell + 1)) / (2 * pi)
        dlth["TE"] = Clte[:, : lmax + 1] * (ell * (ell + 1)) / (2 * pi)
        dlth["ET"] = dlth["TE"]

        chi2 = self.compute_chi2(dlth, params)
        like = -0.5 * chi2

        if self.additional_args.get("add_lowl_tt") is True:
            Cls_tt_2_29 = Cltt[:, 2:30]

            lognorm_tt = utils.planck_lowT_binned_loglike(
                Cls_tt_2_29,
                self.mu_LN_TT,
                self.sig_LN_TT,
                self.lmin_list_TT,
                self.lmax_list_TT,
            )

            like += lognorm_tt

        #if nan or inf return -inf
        if is_jax:
            like = where(isnan(like), -inf, like)
            like = where(isinf(like), -inf, like)
        else:
            if isnan(like) or isinf(like):
                like = -inf
        
        return like

    def set_external_spectra(self, spectra):
        self.external_spectra = spectra
