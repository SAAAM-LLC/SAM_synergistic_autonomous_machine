# setup.py
# Package setup for SAM (Synergistic Autonomous Machine)

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description from README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="synergistic-autonomous-machine",
    version="0.2.0",
    author="SAAAM LLC",
    author_email="contact@saaam.ai",
    description="Synergistic Autonomous Machine: Revolutionary Neural-Linguistic Architecture with Hive Mind Capability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/michaeldubu/SYNERGISTIC_AUTONOMOUS_MACHINE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sam-run=run:main",
            "sam-setup=setup_sam:main",
        ],
    },
)
