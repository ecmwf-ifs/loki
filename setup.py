try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(name='loki',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Experimental Fortran IR to facilitate source-to-source transformations",
      author="Michael Lange",
      packages=find_packages(exclude=['tests']),
      install_requires=['open-fortran-parser'],
      test_requires=['pytest', 'flake8'],
      scripts=['scripts/loki-transform.py'],
)

