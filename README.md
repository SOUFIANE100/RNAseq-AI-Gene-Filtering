# BRCA ML Filtering Pipeline

**Machine Learning-based gene filtering for RNA-seq data**  
Benchmarked against HTSFilter on TCGA Breast Cancer (BRCA) data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This package implements and benchmarks 8 ML classifiers for RNA-seq gene filtering:

| Model | Test AUC | Test F1 | Jaccard vs HTSFilter | Rank |
|-------|----------|---------|----------------------|------|
| LightGBM | 0.9999 | 0.9993 | 0.9986 | 1 |
| XGBoost | 0.9999 | 0.9993 | 0.9989 | 2 |
| Random Forest | 0.9999 | 0.9989 | 0.9978 | 3 |
| MLP | 0.9997 | 0.9976 | 0.9958 | 4 |
| SVM | 0.9997 | 0.9871 | 0.9754 | 5 |
| KNN | 0.9955 | 0.9934 | 0.9875 | 6 |
| CNN | 0.9799 | 0.9845 | 0.9603 | 7 |
| RNN-LSTM | 0.9579 | 0.9767 | 0.9424 | 8 |

**Dataset**: TCGA BRCA RNA-seq (20,530 genes × 1,218 samples)  
**Reference**: HTSFilter (Rau et al., 2013)

---

## Installation

```bash
git clone https://github.com/[username]/brca-ml-filtering.git
cd brca-ml-filtering
pip install -e .

# For deep learning models (CNN, RNN-LSTM):
pip install -e ".[deep]"
```

**R dependencies** (for DEG and enrichment):
```r
install.packages("BiocManager")
BiocManager::install(c("limma", "edgeR", "clusterProfiler", "org.Hs.eg.db"))
install.packages("msigdbr")
```

---

## Quick Start

```python
from brca_ml_filtering import (
    load_expression_matrix,
    compute_gene_features,
    split_train_test,
    train_all_models,
    predict_filtering,
    evaluate_model,
)

# 1. Load data
expr = load_expression_matrix("path/to/counts.csv", log_transform=True, cpm_normalize=True)

# 2. Compute features
features = compute_gene_features(expr)

# 3. Load HTSFilter labels (or generate your own)
import pandas as pd
labels = pd.read_csv("path/to/htsfilter_labels.csv", index_col=0)["label"]

# 4. Split train/test
X_train, X_test, y_train, y_test = split_train_test(features, labels)

# 5. Train models
models = train_all_models(X_train, y_train, model_names=["LightGBM", "XGBoost", "RF"])

# 6. Evaluate
from brca_ml_filtering.models import predict_filtering
from brca_ml_filtering.evaluation import evaluate_model

for name, model in models.items():
    y_pred, y_proba = predict_filtering(model, X_test)
    metrics = evaluate_model(y_test.values, y_pred, y_proba)
    print(f"{name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
```

---

## Repository Structure

```
brca-ml-filtering/
├── brca_ml_filtering/          # Main Python package
│   ├── __init__.py
│   ├── preprocessing.py        # Data loading & feature engineering
│   ├── models.py               # ML model definitions & training
│   ├── evaluation.py           # Metrics: AUC, Jaccard, silhouette
│   └── enrichment.py           # DEG (limma-voom) & pathway enrichment
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_shap_analysis.ipynb
│   ├── 04_filtering_comparison.ipynb
│   └── 05_pathway_enrichment.ipynb
├── scripts/
│   ├── run_pipeline.py         # End-to-end pipeline script
│   └── run_enrichment.R        # R script for DEG + enrichment
├── data/sample/                # Sample data for testing
├── figures/                    # Reproduced paper figures
├── tests/                      # Unit tests
├── setup.py
└── README.md
```

---

## Reproducing Paper Results

```bash
# Full pipeline (requires TCGA BRCA data)
python scripts/run_pipeline.py \
    --expr data/rnaseq_expression_BRCA.csv \
    --meta data/sample_info_BRCA.csv \
    --output results/

# Individual steps
jupyter nbconvert --to notebook --execute notebooks/01_data_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_model_training.ipynb
```

---

## Citation

If you use this package, please cite:

```bibtex
@article{ELATFA,
  title={An Interpretable Machine-Learning Surrogate for HTSFilter: Label-Free RNA-Seq Gene Filtering with Cross-Cancer Validation},
  author={El Atfa, Soufiane},
  journal={MDPI BioMedInformatics},
  year={2026},
  doi={[DOI]}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
