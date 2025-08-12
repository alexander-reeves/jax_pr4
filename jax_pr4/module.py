"""
JAX/NumPy compatibility module for JAX PR4 likelihoods.

This module provides a unified interface for JAX and NumPy operations,
allowing the likelihoods to work seamlessly in both modes.
"""

import os, sys
from jax_pr4.config import get_jax_enabled

global is_jax
is_jax = get_jax_enabled()

if is_jax:
    print("using jax")
    from jax import config
    config.update("jax_enable_x64", True)
    from jax import jit, vmap, disable_jit
    from jax.numpy import (
        savez, load, ndarray, conj, ones, tan, log, log10, 
        logspace, swapaxes, empty, linspace, delete, pi, cos, sin, 
        exp, sqrt, concatenate, linalg, eye, einsum, einsum_path, sum, pad, 
        diag, block, array_equal, meshgrid, geomspace, moveaxis, ones_like, 
        empty_like, real, zeros_like, float32, float64, int32, dot, multiply, 
        add, subtract, unique, isin, newaxis, ix_, transpose, interp, 
        moveaxis, rollaxis, logical_or, tanh, column_stack, power, asarray, 
        sign, matmul, logical_and, stack, all, ndim, ceil, argwhere, cov, 
        nansum, nanmax, identity, triu_indices, repeat, bincount, s_, nan_to_num, 
        copy, take_along_axis, atleast_1d, clip, mod, zeros, array, inf, nan, arange, where, hstack, mean, isnan, isinf, isfinite
    )
    from jax import vmap 
    from jax.numpy import max as module_max
    from jax.numpy import min as module_min
    from jax.numpy.fft import rfft
    from jax.scipy.linalg import block_diag
    from jax.scipy.integrate import trapezoid as trapz
    import jax.numpy as jnp
else:
    import numpy
    from numpy import (
        savez, load, ndarray, conj, ones, tan, log, log10, 
        logspace, swapaxes, empty, linspace, delete, pi, 
        cos, sin, exp, sqrt, trapz, concatenate, linalg, eye, einsum, einsum_path, 
        sum, pad, diag, block, array_equal, meshgrid, trapz, geomspace, 
        moveaxis, ones_like, empty_like, real, zeros_like, float32, float64, 
        int32, dot, multiply, add, subtract, unique, isin, newaxis, ix_, 
        transpose, interp, rollaxis, tanh, column_stack, power, asarray, 
        loadtxt, sign, matmul, logical_and, stack, all, ndim, ceil, 
        argwhere, cov, nansum, nanmax, fill_diagonal, identity, triu_indices, 
        repeat, bincount, s_, nan_to_num, copy, take_along_axis, atleast_1d, 
        clip, mod, zeros, array, inf, nan, arange, where, hstack, mean, isnan, isinf, isfinite
    )
    from numpy import max as module_max
    from numpy import min as module_min
    from numpy.fft import rfft
    from scipy.interpolate import interp1d
    from scipy.linalg import block_diag
    from scipy.special import legendre 