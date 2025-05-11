from setuptools import setup, find_packages

setup(
    name="rfw-analysis",
    version="0.1.0",
    description="Analysis tool for Racial Faces in the Wild dataset",
    author="AI Developer",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "Pillow>=8.0.0",
        "numpy>=1.19.0",
        "rich>=10.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.1.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
