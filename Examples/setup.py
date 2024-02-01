#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:52:47 2023

@author: forootani
"""

from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Phase FPINN',
    version='1.0.0',
    description='A sample package',
    author='Ali Forootani',
    author_email='aliforootani@gmail.com',
    url='https://github.com/Ali-Forootani/Physique-Informed-Neural-Network-for-PDEs',
    packages=[''],
    install_requires=requirements,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.9.7',
    ],
)
