from setuptools import setup, find_packages

setup(
    name='spatial-audio-guidance',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='3D Spatial Scene Understanding and Interactive Audio Guidance for Blind Users',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
        'pydub',
        'gTTS',
        'matplotlib',
        'pytest',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)