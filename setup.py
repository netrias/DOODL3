from setuptools import setup, find_packages

DISTNAME = 'DOODL3'
DESCRIPTION = 'repository for Extracting Global Dynamics of Loss Landscape in Deep Learning Models'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'netrias'
MAINTAINER_EMAIL = 'meslami@netrias.com'

setup(
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    packages=['deep_chaos'] + ['deep_chaos/' + s for s in find_packages('deep_chaos')],
    include_package_data=True,
    install_requires=[
        "keras",
        "matplotlib",
        "graphviz",
        "h5py",
        "numpy",
        "pandas",
        "scipy",
        "seaborn",
        "ipython",
        "sklearn",
        "tensorflow"
    ]
)