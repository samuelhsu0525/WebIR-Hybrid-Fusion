# ML-Based Score Fusion for Document Ranking
These scripts demonstrate how to **fuse** lexical (BM25) and dense (ANCE) retrieval scores using three machine-learning models—**Logistic Regression**, **GBDT**, and **BPR**—to improve ranking quality.

The pipeline has three stages:

1. Generate base retrieval scores  
2. Assemble a training/test set  
3. Train and evaluate fusion models  

## Requirements

- `beir`
- `torch`
- `numpy`
- `joblib`
- `pandas`
- `scikit-learn`
- `xgboost`

## Generate Base Retrieval Scores  
<sup>Scripts: `bm25.py`, `ance.py`</sup>

1. Edit each script to select a BEIR dataset and split:
   ```python
   dataset = "fiqa"
   split   = "test"
   ```
   and specify the output path.

2. Run the scripts:

```bash
python bm25.py
python ance.py
```

Each script will:

* Download and load the dataset.
* Compute scores for each `qid-docid` pair.
* Store them in a nested dictionary:

  ```python
  results[qid][docid] = score
  ```
* Save metrics such as NDCG for later use.

## Assemble Training/Test Data

<sup>Script: `data.py`</sup>

1. In `data.py`, fill in:

   * The filenames of BM25 and ANCE score files.
   * The path to the qrels file (`*.tsv`).
   * The output CSV filename.
2. Run:

   ```bash
   python data.py
   ```

   The script will write a CSV file with five columns:

   | qid | docid | BM25\_score | ANCE\_score | label |
   | --- | ----- | ----------- | ----------- | ----- |
   | …   | …     | …           | …           | 0/1   |

   *`label = 1` if the document is relevant to the query; otherwise `0`.*

## Train & Evaluate Fusion Models

<sup>Scripts: `logistic.py`, `gbdt.py`, `bpr.py`</sup>

Each script defines:

* **`train()`**

  1. Load the training CSV created by `data.py`.
  2. Balance positives and negatives (randomly down-sample negatives).
  3. Train the model and save it using `joblib.dump`.

* **`test()`**

  1. Load the chosen dataset split, testing CSV, and model.
  2. Generate fused scores for each `qid-docid` pair.
  3. Use BEIR’s evaluation utilities to calculate and store NDCG, MAP, Recall, and other metrics.