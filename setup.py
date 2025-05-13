from setuptools import setup, find_packages

setup(
    name="DeepEpitope",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning framework for B-cell epitope prediction using ESM2",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "deepepitope": [
            "models/*.pth",
            "models/*.pkl"
        ]
    },
    install_requires=[
        "torch>=1.10",
        "transformers>=4.26",
        "scikit-learn",
        "pandas",
        "numpy",
        "tqdm",
        "biopython",
        "joblib",
        "openpyxl"
    ],
    entry_points={
        "console_scripts": [
            "DeepEpitope=deepepitope.predictor:main"
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)

