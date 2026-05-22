from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="brca-ml-filtering",
    version="1.0.0",
    author="Soufiane El Atfa",
    description="Machine Learning-based gene filtering for RNA-seq data (TCGA BRCA benchmark)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/brca-ml-filtering",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "scipy>=1.7",
        "lightgbm>=3.3",
        "xgboost>=1.5",
        "shap>=0.40",
        "matplotlib>=3.4",
        "seaborn>=0.11",
    ],
    extras_require={
        "deep": ["tensorflow>=2.8"],
        "dev":  ["pytest", "jupyter", "black", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    keywords="RNA-seq gene filtering machine learning breast cancer TCGA bioinformatics",
)
