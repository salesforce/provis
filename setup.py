# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt', 'r') as reqs:
    requirements = reqs.read().split()

setup(
    name='protein-attention',
    packages=["protein_attention"],
    version='0.0.1',
    install_requires=requirements,
)