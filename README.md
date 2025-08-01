# JAX PR4 - Planck PR4 CMB Likelihoods for JAX

A JAX-compatible implementation of Planck PR4 CMB likelihoods that takes direct Cl inputs rather than cosmological parameters.

## Overview

This package provides JAX-compatible implementations of the Planck PR4 likelihoods:

- **CamSpec**: High-ℓ TT, TE, EE spectra (ℓ ≥ 30)
- **Hillipop**: High-ℓ TT, TE, EE spectra (ℓ ≥ 30) 
- **Lollipop**: Low-ℓ EE, BB, EB spectra (ℓ ≤ 30)

All likelihoods are designed to work with direct Cl inputs rather than cosmological parameters, making them suitable for use with any CMB theory code or emulator.

## Features

- **JAX/NumPy Compatibility**: Seamless switching between JAX and NumPy backends
- **Direct Cl Inputs**: No dependency on cosmological parameter emulators
- **Batch Processing**: Efficient computation for multiple parameter sets

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd jax_pr4

# Install in development mode
pip install -e .
```

### Hillipop large data file

Use the one-liner below to download the HiLLiPoP PR4 v4.2 data bundle and extract **only** the inverse-Fℓℓ′ matrix you need:

```bash
wget -O /tmp/hillipop_TTTEEE_v4.2.tar.gz \
  https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods/planck_2020_hillipop_TTTEEE_v4.2.tar.gz

tar -xzf /tmp/hillipop_TTTEEE_v4.2.tar.gz \
    --wildcards '*/invfll_PR4_v4.2_TTTEEE.fits' \
    -C data/planck_pr4_hillipop --strip-components=2
```

That places `invfll_PR4_v4.2_TTTEEE.fits` inside `data/planck_pr4_hillipop/`, after which the **HiLLiPoP** JAX likelihood should run. 

### Camspec large data file

The high-ℓ Planck PR4 CamSpec data file `like_NPIPE_12.6_unified_cov.bin` is also too large to commit to this repository. 

**Use Cobaya’s built-in downloader**  
    python -m cobaya.download planck_pr4_camspec

   Cobaya will fetch the archive (~3 GB) from the Planck-2020 NERSC mirror, unpack it
   under `~/.local/share/cobaya/planck_pr4_camspec/`, then transfer just that file to `/data/camspec_pr4`

Both archives are large and therefore excluded via `.gitignore`; the commands above avoid checking them into the repository. 

## Dependencies

- jax
- jaxlib
- numpy
- scipy
- astropy
- h5py

## Quick Start

```python
from jax_pr4 import CamSpecPR4, HillipopPR4, LollipopPR4
from jax_pr4.config import set_jax_enabled
import numpy as np

# Initialize likelihoods
camspec = CamSpecPR4()
hillipop = HillipopPR4()
lollipop = LollipopPR4()

# Generate test Cl spectra
ClTT = np.random.randn(1, 2501)  # (batch_size, ell_max+1)
ClTE = np.random.randn(1, 2501)
ClEE = np.random.randn(1, 2501)

# Basic parameters
params = {'A_planck': 1.0}

# Compute likelihoods
loglike_camspec = camspec.compute_like(ClTT, ClTE, ClEE, params)
loglike_hillipop = hillipop.compute_like(ClTT, ClTE, ClEE, params)
loglike_lollipop = lollipop.compute_like(ClEE, params=params)

# Switch to JAX mode
set_jax_enabled(True)
import jax.numpy as jnp
ClTT_jax = jnp.array(ClTT)
loglike_jax = camspec.compute_like(ClTT_jax, ClTE, ClEE, params)
```

## Usage

### Basic Usage

```python
import jax_pr4
from jax_pr4 import CamSpecPR4, HillipopPR4, LollipopPR4
import numpy as np

# Initialize likelihoods
camspec = CamSpecPR4()
hillipop = HillipopPR4()
lollipop = LollipopPR4()

# Generate some test Cl spectra (batch_size=10, ell_max=2500)
batch_size = 10
ell_max = 2500
ClTT = np.random.randn(batch_size, ell_max + 1)
ClTE = np.random.randn(batch_size, ell_max + 1)
ClEE = np.random.randn(batch_size, ell_max + 1)

# Nuisance parameters (see detailed documentation below)
params = {
    'A_planck': 1.0,
    'cal0': 1.0,
    'cal2': 1.0,
    'calTE': 1.0,
    'calEE': 1.0,
}

# Compute log-likelihoods
loglike_camspec = camspec.compute_like(ClTT, ClTE, ClEE, params)
loglike_hillipop = hillipop.compute_like(ClTT, ClTE, ClEE, params)
loglike_lollipop = lollipop.compute_like(ClEE, params=params)

print(f"CamSpec log-likelihood: {loglike_camspec}")
print(f"Hillipop log-likelihood: {loglike_hillipop}")
print(f"Lollipop log-likelihood: {loglike_lollipop}")
```

### JAX Mode

```python
import jax
import jax.numpy as jnp
from jax_pr4.config import set_jax_enabled

# Enable JAX mode
set_jax_enabled(True)

# Now all likelihoods will use JAX arrays
ClTT = jnp.random.randn(batch_size, ell_max + 1)
ClTE = jnp.random.randn(batch_size, ell_max + 1)
ClEE = jnp.random.randn(batch_size, ell_max + 1)

# JAX-compatible computation
loglike = camspec.compute_like(ClTT, ClTE, ClEE, params)
```

### Advanced Configuration

```python
# CamSpec with low-ell components
camspec_config = {
    'add_lowell': True,  # Include low-ell TT and EE lognormal bins
    'with_planck': False,  # Include only low-ell TT
    'apply_planck_temperature_rescaling': True,
    'apply_planck_pol_rescaling': True,
    'binning_matrix': my_binning_matrix,  # For compression
    'binned_covinv': my_binned_covinv,    # Pre-computed binned covariance
}
camspec = CamSpecPR4(additional_args=camspec_config)

# Hillipop with compression and foregrounds
hillipop_config = {
    'add_lowl_tt': True,  # Include low-ell TT lognormal bins
    'hillipop_compression': my_compression_matrix,
}
hillipop = HillipopPR4(additional_args=hillipop_config)
```

## Low-ℓ Lognormal Components

The package supports optional low-ℓ lognormal likelihood components that complement the high-ℓ likelihoods:

### **Low-ℓ TT (Temperature)**
- **Coverage**: ℓ = 2-29
- **Binning**: 2 bins (ℓ = 2-15, ℓ = 16-29)
- **Available in**: CamSpec (`add_lowell=True` or `with_planck=True`) and Hillipop (`add_lowl_tt=True`)
- **Data file**: `lognormal_fit_2bins_TT.txt`

### **Low-ℓ EE (Polarization)**
- **Coverage**: ℓ = 2-29
- **Binning**: 3 bins (ℓ = 2-10, ℓ = 11-20, ℓ = 21-29)
- **Available in**: CamSpec (`add_lowell=True`)
- **Data file**: `lognormal_fit_3bins_EE.txt`

### **Usage Example with Low-ℓ Components**

```python
# CamSpec with both low-ell TT and EE
camspec = CamSpecPR4(additional_args={'add_lowell': True})

# Hillipop with low-ell TT only
hillipop = HillipopPR4(additional_args={'add_lowl_tt': True})

# Compute likelihood with low-ell contributions automatically included
loglike_camspec = camspec.compute_like(ClTT, ClTE, ClEE, params)
loglike_hillipop = hillipop.compute_like(ClTT, ClTE, ClEE, params)
```

## Parameter Documentation

### CamSpec Parameters

**Required Parameters:**
- `A_planck` : float - Overall Planck calibration factor (default: 1.0)
- `cal0` : float - Calibration for 100x100 spectrum (default: 1.0)
- `cal2` : float - Calibration for 217x217 spectrum (default: 1.0)
- `calTE` : float - Calibration for TE spectrum (default: 1.0)
- `calEE` : float - Calibration for EE spectrum (default: 1.0)

**Optional Parameters:**
- `m_t_planck` : float - Temperature rescaling parameter (if apply_planck_temperature_rescaling=True)
- `m_e_planck` : float - Polarization rescaling parameter (if apply_planck_pol_rescaling=True)

**Foreground Parameters (if using foregrounds):**
- `amp_100`, `amp_143`, `amp_217`, `amp_143x217` : float - Dust amplitudes (default: 0.0)
- `n_100`, `n_143`, `n_217`, `n_143x217` : float - Dust spectral indices (default: 1.5)

### Hillipop Parameters

**Required Parameters:**
- `A_planck` : float - Overall Planck calibration factor (default: 1.0)

**Map-specific Calibration Parameters:**
- `cal100A`, `cal100B`, `cal143A`, `cal143B`, `cal217A`, `cal217B` : float - Map calibrations (default: 1.0)

**Polarization Efficiency Parameters:**
- `pe100A`, `pe100B`, `pe143A`, `pe143B`, `pe217A`, `pe217B` : float - Polarization efficiencies (default: 1.0)

**Foreground Parameters (if using foregrounds):**
- `Atsz` : float - Thermal SZ amplitude (default: 0.0)
- `Aksz` : float - Kinetic SZ amplitude (default: 0.0)
- `Acib` : float - CIB amplitude (default: 0.0)
- `AsyncT` : float - Synchrotron temperature amplitude (default: 0.0)
- `AsyncP` : float - Synchrotron polarization amplitude (default: 0.0)
- `AdustT` : float - Dust temperature amplitude (default: 0.0)
- `AdustP` : float - Dust polarization amplitude (default: 0.0)
- `beta_cib` : float - CIB spectral index (default: 1.75)
- `beta_dustT` : float - Dust temperature spectral index (default: 1.5)
- `beta_dustP` : float - Dust polarization spectral index (default: 1.5)
- `xi` : float - SZxCIB correlation parameter (default: 0.0)
- `alpha_tsz` : float - Thermal SZ tilt parameter (only for Battaglia template, default: 0.0)

### Lollipop Parameters

**Required Parameters:**
- `A_planck` : float - Overall Planck calibration factor (default: 1.0)
- `A_act` : float, optional - ACT calibration factor (alternative to A_planck)

**Note:** Either `A_planck` or `A_act` must be provided, but not both.

## Foreground Models

The package includes a comprehensive foreground modeling system for Hillipop, adapted from the original UPanda implementation:

### Available Foreground Models

- **Thermal SZ (tSZ)**: Thermal Sunyaev-Zeldovich effect
- **Kinetic SZ (kSZ)**: Kinetic Sunyaev-Zeldovich effect  
- **CIB**: Clustered Cosmic Infrared Background
- **Dust**: Galactic dust emission
- **Synchrotron**: Galactic synchrotron emission
- **Point Sources**: Radio and dusty point sources
- **SZxCIB**: SZ-CIB correlation
- **Subpixel**: Subpixel effects

### Foreground Usage

```python
# Hillipop with foregrounds
hillipop_config = {
    'tsz_template': 'planck',  # or 'battaglia'
}

hillipop = HillipopPR4("path/to/hillipop/data", hillipop_config)

# Parameters including foregrounds
params = {
    'A_planck': 1.0,
    'cal100A': 1.0, 'cal100B': 1.0,
    'cal143A': 1.0, 'cal143B': 1.0,
    'cal217A': 1.0, 'cal217B': 1.0,
    'pe100A': 1.0, 'pe100B': 1.0,
    'pe143A': 1.0, 'pe143B': 1.0,
    'pe217A': 1.0, 'pe217B': 1.0,
    # Foreground parameters
    'Atsz': 0.1,      # Thermal SZ amplitude
    'Aksz': 0.05,     # Kinetic SZ amplitude
    'Acib': 0.2,      # CIB amplitude
    'beta_cib': 1.75, # CIB spectral index
    'AdustT': 0.3,    # Dust temperature amplitude
    'AdustP': 0.1,    # Dust polarization amplitude
    'beta_dustT': 1.5, # Dust temperature spectral index
    'beta_dustP': 1.5, # Dust polarization spectral index
    'xi': 0.1,        # SZxCIB correlation
}

loglike = hillipop.compute_like(ClTT, ClTE, ClEE, params)
```

## Data Files

Most required data files are included in the package and will be automatically loaded from the `data/` directory. However, due to size limitations, two large files need to be downloaded separately:

1. **Hillipop inverse Fisher matrix**: `data/planck_pr4_hillipop/invfll_PR4_v4.2_TTTEEE.fits` (6.6GB)
2. **CamSpec covariance**: `data/camspec_pr4/like_NPIPE_12.6_unified_cov.bin` (486MB)

These files can be obtained from the original UPanda repository or Planck collaboration data releases.

The package includes:

### CamSpec Data Files
- `like_NPIPE_12.6_unified_spectra.txt`
- `like_NPIPE_12.6_unified_cov.bin`
- `like_NPIPE_12.6_unified_data_ranges.txt`
- Foreground template files (tsz, ksz, dust, etc.)

### Hillipop Data Files
- `binning_v4.2.fits`
- `dl_PR4_v4.2_*.fits` (cross-spectra files)
- `invfll_PR4_v4.2_TTTEEE.fits`
- Foreground model files in `foregrounds/` directory

### Lollipop Data Files
- `cl_lolEB_NPIPE.dat`
- `fiducial_lolEB_planck2018_tensor_lensedCls.dat`
- `clcov_lolEB_NPIPE.fits`

### Low-ℓ Data Files (for low-ell components)
- `lognormal_fit_2bins_TT.txt` - Low-ℓ TT lognormal parameters (2 bins: ℓ=2-15, ℓ=16-29)
- `lognormal_fit_3bins_EE.txt` - Low-ℓ EE lognormal parameters (3 bins: ℓ=2-10, ℓ=11-20, ℓ=21-29)

**Note**: No manual data setup is required - the package handles all data loading automatically.

## Examples

See `example_notebook.ipynb` for a complete demonstration including:
- Computing CMB spectra at Planck best-fit cosmology
- Using both JAX and NumPy modes
- Including low-ℓ lognormal components
- Parameter validation and default handling
- Performance comparisons

## API Reference

### CamSpecPR4

```python
class CamSpecPR4:
    def __init__(self, additional_args=None)
    def compute_like(self, ClTT, ClTE, ClEE, params=None)
    def get_prediction(self, ClTT, ClTE, ClEE, params=None)
```

### HillipopPR4

```python
class HillipopPR4:
    def __init__(self, additional_args=None)
    def compute_like(self, ClTT, ClTE, ClEE, params=None)
    def compute_chi2(self, dlth, params)
```

### LollipopPR4

```python
class LollipopPR4:
    def __init__(self, additional_args=None)
    def compute_like(self, ClEE, ClBB=None, ClEB=None, params=None)
```

### Foreground Models

```python
from jax_pr4.foregrounds_hillipop import fg_list

# Available foreground models
available_models = list(fg_list.keys())
# ['sbpx', 'ps', 'dust', 'dust_model', 'sync', 'ksz', 'ps_radio', 'ps_dusty', 'cib', 'tsz', 'szxcib']
```

## Performance

- **JAX Mode**: Enables automatic differentiation and GPU acceleration
- **Batch Processing**: Efficient computation for multiple parameter sets
- **Compression**: Significant speedup with data compression methods
 