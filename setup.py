# setup.py
from setuptools import setup, find_packages

setup(
    name="nec",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "accelerate",
        "bitsandbytes",
        "unsloth",
        "torch",
        "pandas"
    ],
)
