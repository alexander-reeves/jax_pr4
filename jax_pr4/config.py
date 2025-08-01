# Use a more persistent storage that survives module reloads
import os

def _get_jax_state():
    """Get JAX state from environment variable or default to False"""
    return os.environ.get('JAX_PR4_ENABLED', 'False').lower() == 'true'

def _set_jax_state(value: bool):
    """Set JAX state in environment variable"""
    os.environ['JAX_PR4_ENABLED'] = str(value)

is_jax_enabled = _get_jax_state()  # Default setting

def set_jax_enabled(value: bool):
    global is_jax_enabled
    is_jax_enabled = value
    _set_jax_state(value)
    reload_modules()

def get_jax_enabled():
    return _get_jax_state()

def reload_modules():
    from types import ModuleType
    import sys, importlib
    
    def deep_reload(m: ModuleType, lib='jax_pr4'):
        name = m.__name__  # get the name that is used in sys.modules
        name_ext = name + '.'  # support finding sub modules or packages
        def compare(loaded: str): return (loaded == name) or loaded.startswith(name_ext)
        all_mods = tuple(sys.modules)  # prevent changing iterable while iterating over it
        sub_mods = filter(compare, all_mods)
        for pkg in sorted(sub_mods, key=lambda item: item.count('.'), reverse=True):
            if pkg != '%s.config' % lib:
                importlib.reload(sys.modules[pkg])  # reload packages, beginning with the most deeply nested
        return

    import jax_pr4; deep_reload(jax_pr4)
    # If you have other dependencies to reload, add them here (e.g., import fftlog; deep_reload(fftlog, lib='fftlog'))
    return
