# AmazonHackathon

A Python-based machine learning pipeline for product price prediction using textual product descriptions, structured quantity extraction, and embedding-based feature representations.

This repository contains exploratory analysis, feature engineering, SBERT-based embeddings, TF-IDF modeling, LightGBM training, and several experimental model variants for an Amazon-style product pricing challenge.

## Overview

The project is centered on transforming raw catalog text into useful features for regression. It combines:

- text cleaning and extraction from catalog descriptions
- packed-item / quantity parsing
- numeric feature normalization
- TF-IDF + TruncatedSVD text features
- Sentence-BERT embeddings for semantic similarity and textual context
- LightGBM regression optimized for SMAPE evaluation

This makes it a strong example of a hybrid feature engineering pipeline for structured product data and unstructured text.

## Problem Context

The repository appears to target a product pricing task where each item includes catalog text and metadata such as:

- product description / item text
- values and units
- pack size or quantity
- weight / volume / count information
- price target labels for training

The goal is to predict the price of an item using its textual description and structured metadata, minimizing SMAPE (Symmetric Mean Absolute Percentage Error).

## Repository Structure

- `EDA.py` — exploratory data analysis comparing train/test distribution, variance, PCA, and feature drift.
- `TEXT_EMBEDDING.py` — main training pipeline combining text preprocessing, TF-IDF, SBERT embeddings, and LightGBM regression.
- `Concatenating_npz.py` — utility to concatenate `.npz` feature files.
- `test.py` — small validation / sample script.
- `models/` — alternative model experiments and OCR/integrated model variations.
- `text_embedding/` — smaller experiments for embedding and regression-based models.
- `TEXT_EMBEDDING.html` — lightweight HTML artifact or reference output.
- `Tetx_Embedding_final_no_model.py` — a variant of the embedding pipeline without a final model artifact.
- `SBERT_lgbm_*.py` — LightGBM + SBERT experimentation files.

## Main Workflow

The primary pipeline in `TEXT_EMBEDDING.py` follows this sequence:

1. Clean catalog text using regex extraction rules.
2. Remove irrelevant text blocks and normalize units.
3. Parse pack count, weight, volume, and piece count.
4. Encode text with SentenceTransformer (`all-MiniLM-L6-v2`).
5. Generate TF-IDF features and reduce dimensionality with TruncatedSVD.
6. Concatenate numeric + textual features.
7. Train a LightGBM regressor using validation split.
8. Save model artifacts and embeddings.

## Key Libraries Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- LightGBM
- SentenceTransformers
- Matplotlib
- Seaborn
- Joblib
- tqdm

## Installation

```bash
pip install numpy pandas scipy scikit-learn lightgbm sentence-transformers joblib tqdm matplotlib seaborn
```

## Example Usage

Run the main embedding pipeline:

```bash
python TEXT_EMBEDDING.py
```

Run EDA to inspect train/test feature distributions:

```bash
python EDA.py
```

## Outputs

The pipeline creates:

- saved embeddings in `embeddings_train/` and `embeddings_test/`
- combined feature files in `.npz` format
- model artifact such as `lgb_sbert_model.txt`
- TF-IDF and SVD serialization via joblib
- visualization files for EDA and regression analysis

## Notes

This repository appears to be an experimental and iterative ML project rather than a polished production package. Many scripts are exploratory versions and may contain duplicate or alternative implementations. The strongest central workflow is the architecture implemented in `TEXT_EMBEDDING.py` and the supporting EDA script.

## Suggested Repository Description

`Machine learning project for product price prediction using text embeddings, TF-IDF features, and LightGBM regressor.`

## Suggested Short README Tagline

`Hybrid text + numeric feature modeling for product pricing with SBERT and LightGBM.`

## License

No explicit license file was found in the repository. If this project is intended for public reuse, consider adding an open-source license such as MIT.
