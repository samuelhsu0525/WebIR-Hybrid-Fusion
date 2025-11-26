from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.lexical import BM25Search as BM25

import logging
import pathlib, os
import torch
import argparse
import json

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

try:
    BASE_DIR = pathlib.Path(__file__).parent.absolute()
except NameError:
    BASE_DIR = pathlib.Path(".").absolute()

CACHE_DIR = BASE_DIR / "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def normalize_scores(results):
    normalized_results = {}
    for q_id, doc_scores in results.items():
        if not doc_scores:
            normalized_results[q_id] = {}
            continue
        scores = list(doc_scores.values())
        min_score, max_score = min(scores), max(scores)
        normalized_doc_scores = {}
        if max_score == min_score:
            for doc_id, score in doc_scores.items():
                normalized_doc_scores[doc_id] = 0.5
        else:
            for doc_id, score in doc_scores.items():
                normalized_doc_scores[doc_id] = (score - min_score) / (max_score - min_score)
        normalized_results[q_id] = normalized_doc_scores
    return normalized_results


# --- Fusion Functions ---
def fuse_weighted_sum(norm_results1, norm_results2, weight1, weight2):
    logging.info(f"Applying Weighted Sum Fusion with weight1={weight1}, weight2={weight2}")
    fused_results = {}
    all_query_ids = set(norm_results1.keys()) | set(norm_results2.keys())

    for q_id in all_query_ids:
        scores_m1 = norm_results1.get(q_id, {})
        scores_m2 = norm_results2.get(q_id, {})
        all_doc_ids_for_query = set(scores_m1.keys()) | set(scores_m2.keys())
        current_fused_scores = {}
        for doc_id in all_doc_ids_for_query:
            score1 = scores_m1.get(doc_id, 0.0)
            score2 = scores_m2.get(doc_id, 0.0)
            current_fused_scores[doc_id] = (weight1 * score1) + (weight2 * score2)
        fused_results[q_id] = dict(sorted(current_fused_scores.items(), key=lambda item: item[1], reverse=True))
    return fused_results

def fuse_reciprocal_rank(results1, results2, k_rrf=60):
    logging.info(f"Applying Reciprocal Rank Fusion (RRF) with k_rrf={k_rrf}")
    fused_results = {}
    all_query_ids = set(results1.keys()) | set(results2.keys())

    for q_id in all_query_ids:
        scores_m1 = results1.get(q_id, {})
        scores_m2 = results2.get(q_id, {})

        # Create rank maps
        ranked_docs_m1 = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted(scores_m1.items(), key=lambda item: item[1], reverse=True))}
        ranked_docs_m2 = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted(scores_m2.items(), key=lambda item: item[1], reverse=True))}

        all_doc_ids_for_query = set(scores_m1.keys()) | set(scores_m2.keys())
        current_fused_scores = {}
        for doc_id in all_doc_ids_for_query:
            rrf_score = 0.0
            if doc_id in ranked_docs_m1:
                rrf_score += 1.0 / (k_rrf + ranked_docs_m1[doc_id])
            if doc_id in ranked_docs_m2:
                rrf_score += 1.0 / (k_rrf + ranked_docs_m2[doc_id])
            current_fused_scores[doc_id] = rrf_score
        fused_results[q_id] = dict(sorted(current_fused_scores.items(), key=lambda item: item[1], reverse=True))
    return fused_results

def fuse_max_score(norm_results1, norm_results2):
    logging.info("Applying Max Score Fusion")
    fused_results = {}
    all_query_ids = set(norm_results1.keys()) | set(norm_results2.keys())

    for q_id in all_query_ids:
        scores_m1 = norm_results1.get(q_id, {})
        scores_m2 = norm_results2.get(q_id, {})
        all_doc_ids_for_query = set(scores_m1.keys()) | set(scores_m2.keys())
        current_fused_scores = {}
        for doc_id in all_doc_ids_for_query:
            score1 = scores_m1.get(doc_id, 0.0)
            score2 = scores_m2.get(doc_id, 0.0)
            current_fused_scores[doc_id] = max(score1, score2)
        fused_results[q_id] = dict(sorted(current_fused_scores.items(), key=lambda item: item[1], reverse=True))
    return fused_results

def fuse_min_score(norm_results1, norm_results2):
    logging.info("Applying Min Score Fusion")
    fused_results = {}
    all_query_ids = set(norm_results1.keys()) | set(norm_results2.keys())

    for q_id in all_query_ids:
        scores_m1 = norm_results1.get(q_id, {})
        scores_m2 = norm_results2.get(q_id, {})
        all_doc_ids_for_query = set(scores_m1.keys()) | set(scores_m2.keys())
        current_fused_scores = {}
        for doc_id in all_doc_ids_for_query:
            score1 = scores_m1.get(doc_id, 0.0)
            score2 = scores_m2.get(doc_id, 0.0)
            current_fused_scores[doc_id] = min(score1, score2)
        fused_results[q_id] = dict(sorted(current_fused_scores.items(), key=lambda item: item[1], reverse=True))
    return fused_results

def fuse_product_score(norm_results1, norm_results2):
    logging.info("Applying Product Score Fusion")
    fused_results = {}
    all_query_ids = set(norm_results1.keys()) | set(norm_results2.keys())

    for q_id in all_query_ids:
        scores_m1 = norm_results1.get(q_id, {})
        scores_m2 = norm_results2.get(q_id, {})
        all_doc_ids_for_query = set(scores_m1.keys()) | set(scores_m2.keys())
        current_fused_scores = {}
        for doc_id in all_doc_ids_for_query:
            score1 = scores_m1.get(doc_id, 0.0)
            score2 = scores_m2.get(doc_id, 0.0)
            current_fused_scores[doc_id] = score1 * score2
        fused_results[q_id] = dict(sorted(current_fused_scores.items(), key=lambda item: item[1], reverse=True))
    return fused_results

# --- Main Script ---
def main(args):
    dataset = args.dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(BASE_DIR, "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #### --- Model 1 Setup (Dense) ---
    model1_name = "msmarco-roberta-base-ance-firstp"
    model1_cache_name = model1_name.replace("/", "_")
    cache_file_path1 = CACHE_DIR / f"cached_results_{dataset}_{model1_cache_name}.json"

    logging.info(f"Loading Model 1: {model1_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    model1_beir_model = models.SentenceBERT(model1_name, device=device)
    model1_beir = DRES(model1_beir_model, batch_size=args.batch_size)
    retriever1 = EvaluateRetrieval(model1_beir, score_function="cos_sim")

    results1 = None
    if os.path.exists(cache_file_path1) and not args.force_retrieve:
        logging.info(f"Loading cached results for Model 1 from {cache_file_path1}")
        with open(cache_file_path1, 'r') as f:
            results1 = json.load(f)
    else:
        logging.info("Retrieving results for Model 1...")
        results1 = retriever1.retrieve(corpus, queries)
        logging.info(f"Saving results for Model 1 to {cache_file_path1}")
        with open(cache_file_path1, 'w') as f:
            json.dump(results1, f)

    logging.info("Evaluating Model 1...")
    ndcg1, map1, recall1, precision1 = retriever1.evaluate(qrels, results1, retriever1.k_values)
    mrr1 = retriever1.evaluate_custom(qrels, results1, retriever1.k_values, metric="mrr")
    logging.info(f"Model 1 (NDCG@10): {ndcg1.get('NDCG@10', 0.0):.4f}")


    #### --- Model 2 Setup (BM25) ---
    model2_id = "bm25" # Identifier for BM25
    hostname = "localhost:9201"
    index_name = f"{dataset}-bm25idx"
    initialize = True

    cache_file_path2 = CACHE_DIR / f"cached_results_{dataset}_{model2_id}_{index_name}.json"
    model2_name_for_log = f"BM25 (index: {index_name})" # For logging

    model2_beir = BM25(
        index_name=index_name,
        hostname=hostname,
        initialize=initialize,
        number_of_shards=1,
    )
    retriever2 = EvaluateRetrieval(model2_beir)

    results2 = None
    if os.path.exists(cache_file_path2) and not args.force_retrieve:
        logging.info(f"Loading cached results for Model 2 from {cache_file_path2}")
        with open(cache_file_path2, 'r') as f:
            results2 = json.load(f)
        if initialize:
            model2_beir = BM25(index_name=index_name, hostname=hostname, initialize=False)
            retriever2 = EvaluateRetrieval(model2_beir)

    else:
        if not os.path.exists(cache_file_path2) and initialize == False:
             logging.warning(f"BM25 cache miss for {cache_file_path2} and initialize=False. Indexing may not occur if index doesn't exist.")
        
        logging.info("Retrieving results for Model 2...")
        results2 = retriever2.retrieve(corpus, queries)
        logging.info(f"Saving results for Model 2 to {cache_file_path2}")
        with open(cache_file_path2, 'w') as f:
            json.dump(results2, f)

    logging.info("Evaluating Model 2...")
    ndcg2, map2, recall2, precision2 = retriever2.evaluate(qrels, results2, retriever2.k_values)
    mrr2 = retriever2.evaluate_custom(qrels, results2, retriever2.k_values, metric="mrr")
    logging.info(f"Model 2 ({model2_name_for_log}) (NDCG@10): {ndcg2.get('NDCG@10', 0.0):.4f}")


    #### --- Score Normalization (Important for score-based fusion) ---
    logging.info("Normalizing scores for Model 1...")
    norm_results1 = normalize_scores(results1)
    logging.info("Normalizing scores for Model 2...")
    norm_results2 = normalize_scores(results2)

    #### --- Fusion of Results ---
    logging.info(f"Fusing results using method: {args.fusion_method}")
    fused_results = {}
    if args.fusion_method == "weighted_sum":
        fused_results = fuse_weighted_sum(norm_results1, norm_results2, args.weight1, args.weight2)
    elif args.fusion_method == "rrf":
        fused_results = fuse_reciprocal_rank(results1, results2, k_rrf=args.k_rrf)
    elif args.fusion_method == "max_score":
        fused_results = fuse_max_score(norm_results1, norm_results2)
    elif args.fusion_method == "min_score":
        fused_results = fuse_min_score(norm_results1, norm_results2)
    elif args.fusion_method == "product_score":
        fused_results = fuse_product_score(norm_results1, norm_results2)
    else:
        raise ValueError(f"Unknown fusion method: {args.fusion_method}")

    logging.info("Evaluating Fused Results...")
    k_values_to_evaluate = retriever1.k_values
    ndcg_fused, map_fused, recall_fused, precision_fused = retriever1.evaluate(qrels, fused_results, k_values_to_evaluate)
    mrr_fused = retriever1.evaluate_custom(qrels, fused_results, k_values_to_evaluate, metric="mrr")

    #### --- Detailed Evaluation Summary --- ####
    logging.info("\n\n" + "="*30 + " DETAILED EVALUATION RESULTS " + "="*30)

    def log_detailed_metrics(model_name_log, ndcg, _map, recall, precision, mrr, k_values):
        logging.info(f"\n--- Metrics for {model_name_log} ---")
        for k in k_values:
            logging.info(f"NDCG@{k}: {ndcg.get(f'NDCG@{k}', 0.0):.4f} | "
                         f"MAP@{k}: {_map.get(f'MAP@{k}', 0.0):.4f} | "
                         f"Recall@{k}: {recall.get(f'Recall@{k}', 0.0):.4f} | "
                         f"P@{k}: {precision.get(f'P@{k}', 0.0):.4f} | "
                         f"MRR@{k}: {mrr.get(f'MRR@{k}', 0.0):.4f}")

    log_detailed_metrics(f"Model 1 ({model1_name})", ndcg1, map1, recall1, precision1, mrr1, k_values_to_evaluate)
    log_detailed_metrics(f"Model 2 ({model2_name_for_log})", ndcg2, map2, recall2, precision2, mrr2, k_values_to_evaluate)
    log_detailed_metrics(f"Fused Model ({args.fusion_method})", ndcg_fused, map_fused, recall_fused, precision_fused, mrr_fused, k_values_to_evaluate)

    logging.info("\n" + "="*70)

    #### Save results and runfile for fused model
    results_dir = os.path.join(BASE_DIR, "results", dataset)
    os.makedirs(results_dir, exist_ok=True)

    run_file_name = f"run.fused.{args.fusion_method}"
    if args.fusion_method == "weighted_sum":
        run_file_name += f".w1_{args.weight1}_w2_{args.weight2}"
    elif args.fusion_method == "rrf":
        run_file_name += f".krrf_{args.k_rrf}"
    run_file_name += ".trec"

    results_file_name = f"results.fused.{args.fusion_method}"
    if args.fusion_method == "weighted_sum":
        results_file_name += f".w1_{args.weight1}_w2_{args.weight2}"
    elif args.fusion_method == "rrf":
        results_file_name += f".krrf_{args.k_rrf}"
    results_file_name += ".json"

    run_file_path = os.path.join(results_dir, run_file_name)
    results_file_path = os.path.join(results_dir, results_file_name)

    logging.info(f"Saving fused runfile to: {run_file_path}")
    with open(run_file_path, 'w') as f_out:
        for query_id, doc_scores in fused_results.items():
            # Sort documents by score in descending order for ranking
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs):
                f_out.write(f"{query_id} Q0 {doc_id} {rank+1} {score:.6f} FUSED_{args.fusion_method.upper()}\n")

    logging.info(f"Saving fused evaluation metrics to: {results_file_path}")
    evaluation_summary = {
        "ndcg": ndcg_fused,
        "map": map_fused,
        "recall": recall_fused,
        "precision": precision_fused,
        "mrr": mrr_fused
    }
    with open(results_file_path, 'w') as f_out:
        json.dump(evaluation_summary, f_out, indent=4)


    logging.info("Fusion process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEIR retrieval with result fusion.")
    parser.add_argument("--dataset", type=str, default="trec-covid", help="Name of the BEIR dataset to use (e.g., trec-covid, scifact).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for dense retrieval model.")
    parser.add_argument("--fusion_method", type=str, default="weighted_sum",
                        choices=["weighted_sum", "rrf", "max_score", "min_score", "product_score"],
                        help="Method for fusing results.")
    parser.add_argument("--weight1", type=float, default=0.5, help="Weight for model 1 in weighted_sum fusion.")
    parser.add_argument("--weight2", type=float, default=0.5, help="Weight for model 2 in weighted_sum fusion.")
    parser.add_argument("--k_rrf", type=int, default=60, help="Constant k for RRF fusion.")
    parser.add_argument("--force_retrieve", action="store_true", help="Force re-retrieval even if cached results exist.")

    cli_args = parser.parse_args()

    if cli_args.fusion_method == "weighted_sum":
        logging.info(f"Using weighted_sum with weights: Model1={cli_args.weight1}, Model2={cli_args.weight2}")
    elif cli_args.fusion_method == "rrf":
         logging.info(f"Using RRF with k_rrf: {cli_args.k_rrf}")

    main(cli_args)