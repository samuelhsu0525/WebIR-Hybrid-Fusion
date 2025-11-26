import pandas as pd
import os
import pathlib
import joblib
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir import util
import xgboost as xgb

FEATURES = [
    'BM25_score', 'ANCE_score',
]

def train():
    df = pd.read_csv("")
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0].sample(n=len(pos), random_state=42)
    df = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    X = df[FEATURES].values
    y = df["label"].values

    model = xgb.XGBClassifier(n_estimators=500, max_depth=1)
    model.fit(X, y)

    joblib.dump(model, "")

def test():
    dataset = ""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    _, _, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    tree_model = joblib.load("")
    df = pd.read_csv("")
    df["pred_score"] = tree_model.predict_proba(df[FEATURES].values)[:, 1]

    results = {}
    for qid, group in df.groupby("qid"):
        group = group.copy()
        group["docid"] = group["docid"].astype(str)
        topk = group.sort_values("pred_score", ascending=False).head(1000)
        results[str(qid)] = {row.docid: float(row.pred_score) for _, row in topk.iterrows()}

    retriever = EvaluateRetrieval(
        models.SentenceBERT("Alibaba-NLP/gte-modernbert-base"),
        score_function="cos_sim"
    )

    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values
    )
    mrr = retriever.evaluate_custom(
        qrels, results, retriever.k_values, metric="mrr"
    )

    util.save_results("", ndcg, _map, recall, precision, mrr)

if __name__ == "__main__":
    train()
    test()

