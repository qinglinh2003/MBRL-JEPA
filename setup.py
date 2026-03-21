from setuptools import setup, find_packages
import os

setup(
    name="jepa-mbrl",
    version="0.1.0",
    packages=find_packages(),
    python_requires='>=3.10',
    description="JEPA-style Model-Based RL for Vision-Language Models",
    long_description=open("README.md", "r").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
