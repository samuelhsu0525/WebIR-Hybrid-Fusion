import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import pathlib
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir import util

FEATURES = [
    'BM25_score', 'ANCE_score',
]

class PairwiseDataset(Dataset):
    def __init__(self, df, features):
        self.pairs = []
        for qid, group in df.groupby('qid'):
            pos = group[group['label'] == 1]
            neg = group[group['label'] == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            for _, pos_row in pos.iterrows():
                neg_row = neg.sample(1).iloc[0]
                self.pairs.append((
                    pos_row[features].values.astype(np.float32),
                    neg_row[features].values.astype(np.float32)
                ))
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

class LinearRanker(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

def train():
    df = pd.read_csv("")
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, "")

    dataset = PairwiseDataset(df, FEATURES)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LinearRanker(len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(50):
        total_loss = 0
        for pos, neg in dataloader:
            pos = pos.float()
            neg = neg.float()
            pos_score = model(pos)
            neg_score = model(neg)
            loss = bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "")

def test():
    dataset = ""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    _, _, qrels = GenericDataLoader(data_folder=data_path).load(split="")
    
    scaler = joblib.load("")
    model = LinearRanker(len(FEATURES))
    model.load_state_dict(torch.load("", map_location="cpu"))
    model.eval()

    df = pd.read_csv("")
    df_norm = df.copy()
    df_norm[FEATURES] = scaler.transform(df_norm[FEATURES])

    with torch.no_grad():
        feats = torch.tensor(df_norm[FEATURES].values, dtype=torch.float32)
        scores = model(feats).numpy()
    df["pred_score"] = scores

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

if __name__ == '__main__':
    train()
    test()
