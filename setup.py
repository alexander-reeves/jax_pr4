from setuptools import setup, find_packages

setup(
    name="jax_pr4",
    version="0.1.0",
    description="JAX-ified Planck PR4 CMB likelihoods",
    author="Alexander Reeves",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
    ],
    python_requires=">=3.8",
) 