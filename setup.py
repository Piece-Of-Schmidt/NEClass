# setup.py
from setuptools import setup, find_packages
import os

setup(
    name="neclass",
    version="0.1.0",
    author="Tobias Schmidt",
    description="Context-Dependent Named Entity Classification using LLMs",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.40.1",
        "pandas",
        "tqdm"
    ],
    python_requires='>=3.9',
)
