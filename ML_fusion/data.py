import pandas as pd
import os
import pickle
import pathlib

results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "BM25_results")
with open(os.path.join(results_dir, ""), "rb") as fIn:
    BM25_results = pickle.load(fIn)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "ANCE_results")
with open(os.path.join(results_dir, ""), "rb") as fIn:
    ANCE_results = pickle.load(fIn)
score_dicts = [BM25_results, ANCE_results]

label_df = pd.read_csv("", sep='\t', header=0)
label_set = set(zip(label_df['query-id'].astype(str), label_df['corpus-id'].astype(str)))

data = []
for qid in score_dicts[0].keys():
    print(f"\rProcessing qid: {qid}", end='', flush=True)
    docids = set()
    for d in score_dicts:
        docids.update(d.get(qid, {}).keys())
    for docid in docids:
        features = []
        for d in score_dicts:
            min_val = min(d.get(qid, {}).values()) if d.get(qid, {}) else 0
            features.append(d.get(qid, {}).get(docid, min_val))
        label = 1 if (str(qid), str(docid)) in label_set else 0
        data.append([qid, docid] + features + [label])

colnames = ["qid", "docid", "BM25_score", "ANCE_score", "label"]

df = pd.DataFrame(data, columns=colnames)
df.to_csv("", index=False)
