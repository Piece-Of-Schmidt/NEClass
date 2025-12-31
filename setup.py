# setup.py
from setuptools import setup, find_packages
import os

setup(
    name="neclass",
    version="0.1.0",
    author="Tobias Schmidt",
    description="Context-Dependent Named Entity Classification using LLMs",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.40.1",
        "accelerate",
        "bitsandbytes",
        "pandas",
    ],
    extras_require={
        "gpu": ["unsloth", "torch"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.9',
)
