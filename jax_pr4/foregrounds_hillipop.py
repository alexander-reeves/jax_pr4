"""
Foreground models for Hillipop likelihood

This module provides foreground models for the Planck PR4 Hillipop likelihood,
adapted from the original UPanda implementation for JAX compatibility.
"""

from .module import *
import numpy as np
import itertools
import os

# Physical constants
t_cmb = 2.72548
k_b = 1.3806503e-23
h_pl = 6.626068e-34

class fgmodel:
    """
    Base class for foreground models for the Hillipop likelihood.
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters 
    for all correlation of frequencies.
    """

    # Reference frequency for residuals amplitudes
    f0 = 143

    # Planck effective frequencies
    fsz = {100: 100.24, 143: 143, 217: 222.044}
    fdust = {
        100: 105.2,
        143: 147.5,
        217: 228.1,
        353: 370.5,
    }  # alpha=4 from [Planck 2013 IX]
    fcib = fdust
    fsyn = {100: 100, 143: 143, 217: 217}
    fradio = {100: 100.4, 143: 140.5, 217: 218.6}

    def _f_tsz(self, freq):
        """Thermal SZ frequency dependence."""
        # Freq in GHz
        nu = freq * 1e9
        xx = h_pl * nu / (k_b * t_cmb)
        return xx * (1 / tanh(xx / 2.0)) - 4

    def _f_Planck(self, f, T):
        """Planck function."""
        # Freq in GHz
        nu = f * 1e9
        xx = h_pl * nu / (k_b * T)
        return (nu**3.0) / (exp(xx) - 1.0)

    def _dBdT(self, f):
        """Temperature antenna conversion."""
        # Freq in GHz
        nu = f * 1e9
        xx = h_pl * nu / (k_b * t_cmb)
        return (nu) ** 4 * exp(xx) / (exp(xx) - 1.0) ** 2.0

    def _tszRatio(self, f, f0):
        """Thermal SZ frequency ratio."""
        return self._f_tsz(f) / self._f_tsz(f0)

    def _cibRatio(self, f, f0, beta=1.75, T=25):
        """CIB frequency ratio."""
        return (
            power((f / f0), beta)
            * (self._f_Planck(f, T) / self._f_Planck(f0, T))
            / (self._dBdT(f) / self._dBdT(f0))
        )

    def _dustRatio(self, f, f0, beta=1.5, T=19.6):
        """Dust frequency ratio."""
        return (
            (f / f0) ** beta
            * (self._f_Planck(f, T) / self._f_Planck(f0, T))
            / (self._dBdT(f) / self._dBdT(f0))
        )

    def _radioRatio(self, f, f0, beta=-0.7):
        """Radio frequency ratio."""
        return (f / f0) ** beta / (self._dBdT(f) / self._dBdT(f0))

    def _syncRatio(self, f, f0, beta=-0.7):
        """Synchrotron frequency ratio."""
        return (f / f0) ** beta / (self._dBdT(f) / self._dBdT(f0))

    def __init__(self, lmax, freqs, mode="TT", auto=False, **kwargs):
        """Initialize foreground model."""
        self.mode = mode
        self.lmax = lmax
        self.freqs = freqs
        self.name = None

        lmax = 2500

        ell = arange(lmax + 1)
        self.ll2pi = ell * (ell + 1) / (3000 * 3001)

        # Build the list of cross frequencies
        self._cross_frequencies = list(
            itertools.combinations_with_replacement(freqs, 2)
            if auto
            else itertools.combinations(freqs, 2)
        )

    def _gen_dl_powerlaw(self, alpha, lnorm=3000):
        """Generate power-law Dl template."""
        lmax = self.lmax if lnorm is None else max(self.lmax, lnorm)
        ell = arange(2, lmax + 1)

        template = zeros(lmax + 1)
        template[array(ell, int)] = ell * (ell + 1) / 2 / pi * ell ** (alpha)

        # normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[: self.lmax + 1]

    def _read_dl_template(self, filename, lsize=0, lnorm=3000):
        """Read FG template (in Dl, muK^2)."""
        # read dl template
        ell, data = np.loadtxt(filename, unpack=True)
        ell = array(ell, int)

        template = array(data)

        # normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[: 2500 + 1]

    def compute_dl(self, pars):
        """Return spectra model for each cross-spectra."""
        pass


# Subpixel effect
class subpix(fgmodel):
    """Subpixel effect foreground model."""
    
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "SubPixel"
        self.fwhm = {100: 9.68, 143: 7.30, 217: 5.02}  # arcmin

    def compute_dl(self, pars):
        """Compute subpixel effect Dl."""
        def _bl(fwhm):
            sigma = deg2rad(fwhm / 60.0) / sqrt(8.0 * log(2.0))
            ell = arange(self.lmax + 1)
            return exp(-0.5 * ell * (ell + 1) * sigma**2)

        dl_sbpx = []
        for f1, f2 in self._cross_frequencies:
            pxl = self.ll2pi / _bl(self.fwhm[f1]) / _bl(self.fwhm[f2])
            dl_sbpx.append(pars["Asbpx_{}x{}".format(f1, f2)] * pxl / pxl[2500])

        if self.mode == "TT":
            return array(dl_sbpx)
        else:
            return 0.0


# Point Sources
class ps(fgmodel):
    """Point sources foreground model."""
    
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS"

    def compute_dl(self, pars):
        """Compute point sources Dl."""
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            dl_ps.append(
                array(pars["Aps_{}x{}".format(f1, f2)])[:, newaxis]
                * self.ll2pi[newaxis, :]
            )

        if self.mode == "TT":
            return array(dl_ps)
        else:
            return 0.0


# Radio Point Sources (v**alpha)
class ps_radio(fgmodel):
    """Radio point sources foreground model."""
    
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS radio"

    def compute_dl(self, pars):
        """Compute radio point sources Dl."""
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi[newaxis, :]
                * self._radioRatio(self.fradio[f1], self.f0, beta=pars["beta_radio"])[
                    :, newaxis
                ]
                * self._radioRatio(self.fradio[f2], self.f0, beta=pars["beta_radio"])[
                    :, newaxis
                ]
            )

        if self.mode == "TT":
            return array(pars["Aradio"])[newaxis, :, newaxis] * array(dl)
        else:
            return 0.0


# Infrared Point Sources
class ps_dusty(fgmodel):
    """Dusty point sources foreground model."""
    
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS dusty"

    def compute_dl(self, pars):
        """Compute dusty point sources Dl."""
        beta_dusty = array(pars["beta_cib"])
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi[newaxis, :]
                * self._cibRatio(self.fcib[f1], self.f0, beta=beta_dusty)[:, newaxis]
                * self._cibRatio(self.fcib[f2], self.f0, beta=beta_dusty)[:, newaxis]
            )

        if self.mode == "TT":
            return array(pars["Adusty"])[newaxis, :, newaxis] * array(dl)
        else:
            return 0.0


# Galactic Dust
class dust(fgmodel):
    """Galactic dust foreground model."""
    
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust"

        self.dlg = []
        hdr = ["ell", "100x100", "100x143", "100x217", "143x143", "143x217", "217x217"]
        data = np.loadtxt(f"{filename}_{mode}.txt").T
        ell = array(data[0], int)
        for f1, f2 in self._cross_frequencies:
            tmpl = zeros(max(ell) + 1)
            tmpl[ell] = data[hdr.index(f"{f1}x{f2}")]
            self.dlg.append(tmpl[: lmax + 1])

        self.dlg = array(self.dlg)

    def compute_dl(self, pars):
        """Compute dust Dl."""
        if self.mode == "TT":
            A = B = {100: pars["Ad100T"], 143: pars["Ad143T"], 217: pars["Ad217T"]}
        if self.mode == "EE":
            A = B = {100: pars["Ad100P"], 143: pars["Ad143P"], 217: pars["Ad217P"]}
        if self.mode == "TE":
            A = {100: pars["Ad100T"], 143: pars["Ad143T"], 217: pars["Ad217T"]}
            B = {100: pars["Ad100P"], 143: pars["Ad143P"], 217: pars["Ad217P"]}
        if self.mode == "ET":
            A = {100: pars["Ad100P"], 143: pars["Ad143P"], 217: pars["Ad217P"]}
            B = {100: pars["Ad100T"], 143: pars["Ad143T"], 217: pars["Ad217T"]}

        Ad = [A[f1] * B[f2] for f1, f2 in self._cross_frequencies]

        return array(Ad)[:, None] * self.dlg


class dust_model(fgmodel):
    """Dust model foreground."""
    
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust model"

        self.dlg = []
        hdr = ["ell", "100x100", "100x143", "100x217", "143x143", "143x217", "217x217"]
        data = np.loadtxt(f"{filename}_{mode}.txt").T
        ell = array(data[0], int)
        for f1, f2 in self._cross_frequencies:
            tmpl = array(data[hdr.index(f"{f1}x{f2}")])
            self.dlg.append(tmpl[: 2500 + 1])
        self.dlg = array(self.dlg)

    def compute_dl(self, pars):
        """Compute dust model Dl."""
        mode = self.mode
        beta1 = array(pars[f"beta_dust{'T' if mode[0] == 'T' else 'P'}"])
        beta2 = array(pars[f"beta_dust{'T' if mode[1] == 'T' else 'P'}"])
        ad1 = array(pars[f"Adust{'T' if mode[0] == 'T' else 'P'}"])
        ad2 = array(pars[f"Adust{'T' if mode[1] == 'T' else 'P'}"])

        dl = []

        for xf, (f1, f2) in enumerate(self._cross_frequencies):
            dust_ratio1 = self._dustRatio(self.fdust[f1], self.fdust[353], beta=beta1)
            dust_ratio2 = self._dustRatio(self.fdust[f2], self.fdust[353], beta=beta2)

            product = (
                ad1[:, newaxis]
                * ad2[:, newaxis]
                * dust_ratio1[:, newaxis]
                * dust_ratio2[:, newaxis]
                * self.dlg[xf][newaxis, :]
            )

            dl.append(product)

        return array(dl)


# Synchrotron model
class sync_model(fgmodel):
    """Synchrotron foreground model."""
    
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Synchrotron"

        # check effective freqs
        for f in freqs:
            if f not in self.fsyn:
                raise ValueError(f"Missing SYNC effective frequency for {f}")

        alpha_syn = -2.5  # Cl template power-law TBC
        self.dl_syn = self._gen_dl_powerlaw(alpha_syn, lnorm=100)
        self.beta_syn = -0.7

    def compute_dl(self, pars):
        """Compute synchrotron Dl."""
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.dl_syn[newaxis, :]
                * self._syncRatio(self.fsyn[f1], self.f0, beta=self.beta_syn)[
                    :, newaxis
                ]
                * self._syncRatio(self.fsyn[f2], self.f0, beta=self.beta_syn)[
                    :, newaxis
                ]
            )
        if self.mode == "TT":
            return array(pars["AsyncT"])[newaxis, :, newaxis] * array(dl)
        elif self.mode == "EE":
            return array(pars["AsyncP"])[newaxis, :, newaxis] * array(dl)
        else:
            return 0.0


# CIB model
class cib_model(fgmodel):
    """Clustered CIB foreground model."""
    
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "clustered CIB"

        # check effective freqs
        for f in freqs:
            if f not in self.fcib:
                raise ValueError(f"Missing CIB effective frequency for {f}")

        if filename is None:
            alpha_cib = -1.3
            self.dl_cib = self._gen_dl_powerlaw(alpha_cib)
        else:
            self.dl_cib = self._read_dl_template(filename, lsize=13000)

    def compute_dl(self, pars):
        """Compute CIB Dl."""
        dl = []

        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.dl_cib[newaxis, :]
                * self._cibRatio(self.fcib[f1], self.f0, beta=pars["beta_cib"])[
                    :, newaxis
                ]
                * self._cibRatio(self.fcib[f2], self.f0, beta=pars["beta_cib"])[
                    :, newaxis
                ]
            )
        if self.mode == "TT":
            return array(pars["Acib"])[newaxis, :, newaxis] * array(dl)
        else:
            return 0.0


# Thermal SZ model
class tsz_model(fgmodel):
    """
    Thermal-SZ power-spectrum foreground.

    template = "planck"   → Planck EM12 shape, ν0 = 143 GHz
    """
    
    def __init__(self, lmax, freqs,
                 filename="",
                 *, template="planck",
                 data_folder=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "tSZ"
        self.template = template.lower()

        # Reference frequency ν0
        self.nu0 = 143   # [GHz]

        # Make sure effective ν are defined
        for f in freqs:
            if f not in self.fsz:
                raise ValueError(f"Missing SZ effective frequency for {f} GHz")
        
        Dl_raw = self._read_dl_template(filename, lsize=10000)[:lmax+1]

        # ℓ array & optional tilt
        self.ells = arange(lmax + 1)
        self.base_tsz = Dl_raw            # shape-only template

        # Pre-compute freq-scaled spectra
        self.dl_sz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_sz.append(
                self.base_tsz *
                self._tszRatio(self.fsz[f1], self.nu0) *
                self._tszRatio(self.fsz[f2], self.nu0)
            )
        self.dl_sz = asarray(self.dl_sz)

    def compute_dl(self, pars):
        """
        Atsz       – μK² at ℓ=3000 and ν0  (ν0=143 GHz for Planck)
        """
        Dl = asarray(pars["Atsz"])[None, :, None] * self.dl_sz[:, None, :]

        return Dl


# kSZ
class ksz_model(fgmodel):
    """Kinetic SZ foreground model."""
    
    def __init__(self, lmax, freqs, filename="", mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "kSZ"

        # read Dl template (normalized at l=3000)
        ksztmpl = self._read_dl_template(filename, lsize=10001)

        self.dl_ksz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksztmpl[: 2500 + 1])
        self.dl_ksz = array(self.dl_ksz)

    def compute_dl(self, pars):
        """Compute kSZ Dl."""
        if self.mode == "TT":
            return array(pars["Aksz"])[newaxis, :, newaxis] * self.dl_ksz[:, newaxis, :]
        else:
            return 0.0


# SZxCIB model
class szxcib_model(fgmodel):
    """SZxCIB correlation foreground model."""
    
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False, **kwargs):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "SZxCIB"

        # check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz:
                raise ValueError(f"Missing SZ effective frequency for {f}")

        # check effective freqs for dust
        for f in freqs:
            if f not in self.fcib:
                raise ValueError(f"Missing Dust effective frequency for {f}")

        self._is_template = filename
        if self._is_template:
            self.x_tmpl = self._read_dl_template(filename, lsize=9999)
        elif "filenames" in kwargs:
            self.x_tmpl = self._read_dl_template(
                kwargs["filenames"][0]
            ) * self._read_dl_template(kwargs["filenames"][1], lsize=9999)
        else:
            raise ValueError("Missing template for SZxCIB")

    def compute_dl(self, pars):
        """Compute SZxCIB Dl."""
        dl_szxcib = []
        for f1, f2 in self._cross_frequencies:
            dl_szxcib.append(
                self.x_tmpl[newaxis, :]
                * (
                    self._tszRatio(self.fsz[f2], self.f0)
                    * self._cibRatio(
                        self.fcib[f1], self.f0, beta=array(pars["beta_cib"])
                    )[:, newaxis]
                    + self._tszRatio(self.fsz[f1], self.f0)
                    * self._cibRatio(
                        self.fcib[f2], self.f0, beta=array(pars["beta_cib"])
                    )[:, newaxis]
                )
            )

        if self.mode == "TT":
            return (
                -1.0
                * array(pars["xi"])[newaxis, :, newaxis]
                * sqrt(array(pars["Acib"]) * array(pars["Atsz"]))[newaxis, :, newaxis]
                * array(dl_szxcib)
            )
        else:
            return 0.0


# List of available foreground models
fg_list = {
    "sbpx": subpix,
    "ps": ps,
    "dust": dust,
    "dust_model": dust_model,
    "sync": sync_model,
    "ksz": ksz_model,
    "ps_radio": ps_radio,
    "ps_dusty": ps_dusty,
    "cib": cib_model,
    "tsz": tsz_model,
    "szxcib": szxcib_model,
} 