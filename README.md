# Enhancing Searching Performance with Hybrid Lexical-Dense Retrieval Models

This repository contains the implementation and evaluation of various fusion strategies for Information Retrieval, combining **Lexical (BM25)** and **Dense (ANCE)** retrieval models. The project explores both heuristic (Non-ML) and Machine Learning-based fusion techniques to improve ranking performance across different domains.

## Introduction
Traditional retrieval methods (like BM25) rely on exact keyword matching, while dense retrieval models (like ANCE) capture semantic meaning. This project aims to combine the strengths of both paradigms using score fusion strategies. We evaluate performance on **FiQA-2018** (Financial) and **TREC-COVID** (Medical/Zero-shot) datasets using the [BEIR Benchmark](https://github.com/beir-cellar/beir).

### BEIR
This project relies on the **BEIR** framework. To set up the environment, please follow the instructions at https://github.com/beir-cellar/beir.

### Supported Fusion Methods
| Category | Methods | Description |
| :--- | :--- | :--- |
| **Non-ML** | **Weighted Sum** | Linear combination: $\alpha \cdot S_{lex} + (1-\alpha) \cdot S_{dense}$ |
| | **Min / Max Score** | Conservative vs. aggressive selection of scores. |
| | **Product Score** | Multiplies scores to emphasize mutual agreement. |
| | **RRF** | Reciprocal Rank Fusion based on rank positions. |
| **ML-Based** | **Logistic Regression** | Supervised binary classification for relevance. |
| | **GBDT** | Gradient Boosting Decision Trees (XGBoost). |
| | **BPR** | Bayesian Personalized Ranking for pairwise optimization. |

## Directory Structure

```text
.
├── beir/                  # Reference to BEIR benchmark
├── ML_fusion/             # Machine Learning-based fusion pipeline
│   ├── ance.py            # Generates ANCE scores
│   ├── bm25.py            # Generates BM25 scores
│   ├── data.py            # Prepares training data (merges scores + qrels)
│   ├── logistic.py        # Logistic Regression Training/Testing
│   ├── gbdt.py            # GBDT Training/Testing
│   └── bpr.py             # BPR Training/Testing
└── non_ML_fusion/         # Heuristic fusion pipeline
    ├── fusion.py          # Main script for Non-ML fusion
    └── cache/             # Stores retrieved results to avoid re-running
```

## Usage

### 1. Non-ML Fusion Methods
The `non_ML_fusion` directory contains a streamlined script that handles retrieval, normalization, and fusion in one go.

**Arguments:**
* `--dataset`: The BEIR dataset name (e.g., `trec-covid`, `fiqa`, `scifact`).
* `--fusion_method`: Choose from `weighted_sum`, `rrf`, `max_score`, `min_score`, `product_score`.
* `--weight1`, `--weight2`: Weights for weighted sum (default: 0.5).
* `--k_rrf`: Constant for RRF (default: 60).

**Example:**
```bash
# Run Product Score fusion on TREC-COVID
python non_ML_fusion/fusion.py --dataset trec-covid --fusion_method product_score

# Run Weighted Sum on FiQA
python non_ML_fusion/fusion.py --dataset fiqa --fusion_method weighted_sum --weight1 0.3 --weight2 0.7
```

### 2. ML-Based Fusion Methods
The `ML_fusion` pipeline requires a multi-step process: generating base scores, preparing the dataset, and then training the model.

**Step 1: Generate Base Scores**
Modify `dataset` variable in `ML_fusion/bm25.py` and `ML_fusion/ance.py` to your desired dataset (e.g., "fiqa"), then run:
```bash
cd ML_fusion
python bm25.py
python ance.py
```
This generates score files in `BM25_results/` and `ANCE_results/`.

**Step 2: Assemble Training Data**
Run `data.py` to merge the scores with the ground truth labels (`qrels`).
```bash
python data.py
```
This creates a CSV file containing features (BM25_score, ANCE_score) and labels.


**Step 3: Train & Evaluate Run specific model scripts to train on the generated data and evaluate on the test set.**
```bash
python bpr.py       # Train and Test BPR
python gbdt.py      # Train and Test GBDT
python logistic.py  # Train and Test Logistic Regression
```

## Experimental Results

Performance Comparison (@10) on **FiQA-2018** (In-domain) and **TREC-COVID** (Out-of-domain).

| Method | FiQA NDCG@10 | FiQA MAP@10 | TREC-COVID NDCG@10 | TREC-COVID MAP@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Baselines** | | | | |
| BM25 | 0.2536 | 0.1910 | 0.6880 | 0.0170 |
| ANCE | 0.2709 | 0.2081 | 0.5511 | 0.0116 |
| **Non-ML Fusion** | | | | |
| Weighted Sum | 0.3295 | 0.2558 | 0.7383 | 0.0182 |
| Min Score | 0.3033 | 0.2367 | **0.7668** | **0.0186** |
| Product Score | 0.3173 | 0.2483 | 0.7593 | 0.0191 |
| RRF | 0.3112 | 0.2394 | 0.7503 | 0.0186 |
| **ML Fusion** | | | | |
| Logistic | 0.3198 | 0.2497 | 0.7311 | 0.0182 |
| GBDT | 0.3181 | 0.2473 | 0.7238 | 0.0171 |
| **BPR** | **0.3431** | **0.2721** | 0.7495 | 0.0185 |

### Conclusion
1.  **In-Domain (FiQA):** ML-based methods, particularly **BPR**, outperform heuristics by effectively learning pairwise preferences from training data.
2.  **Zero-Shot (TREC-COVID):** Simple heuristics like **Min Score** and **Product Score** generalize better. ML methods may suffer from overfitting to the training domain (FiQA) when applied to a significantly different distribution (COVID-19 data).

## References
* [1] Thakur, N., et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models."
* [2] "Sparse Meets Dense: A Hybrid Approach to Enhance Scientific Document Retrieval."
* [3] "An Analysis of Fusion Functions for Hybrid Retrieval."
