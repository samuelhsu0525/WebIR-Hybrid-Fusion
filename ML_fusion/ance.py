from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import pickle
import logging
import pathlib, os

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = ""
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="")

model = DRES(models.SentenceBERT("msmarco-roberta-base-ance-firstp"))

retriever = EvaluateRetrieval(model, score_function="dot")
results = retriever.retrieve(corpus, queries)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "ANCE_results")
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, ""), "wb") as fOut:
    pickle.dump(results, fOut)
util.save_results(os.path.join(results_dir, ""), ndcg, _map, recall, precision, mrr)