# setup.py
from setuptools import setup, find_packages

setup(
    name="neclass",
    version="0.1.0",
    author="Tobias Schmidt",
    description="Context-Dependent Named Entity Classification using LLMs",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers>=4.45.0"
    ],
    python_requires='>=3.9',
)
