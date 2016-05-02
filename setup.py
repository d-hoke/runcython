#!/usr/bin/env python
from setuptools import setup
import os
import sys

setup(
    name='runcython',
    version='0.2.5',
    description='compile and run cython in one line',
    license='MIT',
    url='https://github.com/Russell91/runcython',
    long_description='https://github.com/Russell91/runcython',
    install_requires='Cython>=0.10',
    scripts = ['runcython', 'runcython++', 'makecython', 'makecython++'],
)
