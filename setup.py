from distutils.core import setup

from setuptools import find_packages

setup(
    name='rlkit_pmoe',
    version='0.2.1dev',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
)