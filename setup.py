# setup.py
from setuptools import setup, find_packages

setup(
    name="neclass", 
    version="0.1.0",
    author="Tobias Schmidt / TU Dortmund University",
    description="Named Entity Classification (NECLass) using Fine-Tuned LLMs",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Piece-Of-Schmidt/NEClass",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.9',
    install_requires=[
        "transformers>=4.40.1",
        "accelerate",
        "bitsandbytes",
        "pandas",
        "torch",
    ],
)
