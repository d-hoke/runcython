#!/usr/bin/env python
from setuptools import setup
import os
import sys

setup(
    name='runcython',
    version='0.1.9',
    description='compile and run cython in one line',
    license='MIT',
    url='https://github.com/Russell91/runcython',
    long_description='https://github.com/Russell91/runcython',
    install_requires='cython',
    scripts = ['runcython', 'runcython++', 'makecython', 'makecython++'],
)
