try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(name='ecir',
      version='0.01-alpha',
      description="Experiment Fortran parser/IR.",
      author="Michael Lange",
      packages=find_packages(exclude=['tests']),
      install_requires=['open-fortran-parser'],
      test_requires=['pytest', 'flake8']
)

