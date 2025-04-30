from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="spatial-guidance",
    version="0.1.0",
    author="DLVC-Group-04",
    description="DLVC-SoSe25 - 3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in (Path(__file__).parent / "requirements.txt").open()
        if line.strip() and not line.startswith("#")
    ],
)
