import os
from pathlib import Path
from typing import Dict

import math
import optuna
import pandas as pd
from catboost import utils
from tqdm import tqdm

import solution

DATA_DIR = Path("./data/text-ranking-homework-vk-ir-fall-2024")


def load_true_rels(true_rels_path) -> Dict[str, Dict[str, int]]:
    true_rels = pd.read_csv(true_rels_path, sep=' ', header=None)
    query_to_doc_rels = {}

    for row in true_rels.itertuples():
        query_id = row[1]
        doc_id = row[3]
        rel = row[4]
        if query_id not in query_to_doc_rels:
            query_to_doc_rels[query_id] = {}
        query_to_doc_rels[query_id][doc_id] = rel

    return query_to_doc_rels


def dcg(y, k=10):
    r = 0.
    for i, y_i in enumerate(y):
        p = i + 1
        r += (2 ** y_i - 1) / math.log(1 + p, 2)
        if p == k:
            break
    return r


def ndcg(y, k=10):
    if len(y) == 0:
        return 0

    dcg_k = dcg(y, k=k)

    max_dcg = dcg(sorted(y, reverse=True), k=k)

    if max_dcg == 0:
        return 1.

    return dcg_k / max_dcg


def calculate_ndcgk(scores: dict[str, list[(str, float)]], true_rels: dict[str, dict[str, int]], k) -> float:
    true_rels_arr = []
    predict_rels = []
    groups = []

    for query_id, docs in scores.items():
        docs = sorted(docs, key=lambda x: x[1], reverse=True)
        for i, (doc_id, _) in enumerate(docs):
            true_rels_arr.append(true_rels[query_id][doc_id])
            predict_rels.append(len(docs) - i)
            groups.append(query_id)

    return utils.eval_metric(true_rels_arr, predict_rels, f'NDCG:top={k};type=Exp', group_id=groups)[0]


def convert_docdev_queries_to_submission_format(file_name):
    with open(DATA_DIR / "vkmarco-docdev-qrels.tsv") as f:
        if os.path.exists(DATA_DIR / file_name):
            os.remove(DATA_DIR / file_name)
        with open(DATA_DIR / file_name, "a") as submission_f:
            submission_f.write("QueryId,DocumentId\n")
            for line in tqdm(f, total=153372, desc="Converting queries to submission format"):
                query_id, _, doc_id, _ = line.strip().split()
                submission_f.write(f"{query_id},{doc_id}\n")


def main2():
    true_rels = load_true_rels(DATA_DIR / "vkmarco-docdev-qrels.tsv")

    tr = solution.TextRanking(
        DATA_DIR,
        "vkmarco-docdev-qrels_sub.csv",
        "vkmarco-docdev-queries"

    )

    scores = tr.calculate_scores(ver='local')

    print(calculate_ndcgk(scores, true_rels, 10))


def objective(trial: optuna.Trial, tr, true_rels, k: int = 10) -> float:
    config = solution.BM25RankerConfig(
        k1=trial.suggest_float('k1', 0.1, 2.0),
        b=trial.suggest_float('b', 0.1, 1.0),
        nc=trial.suggest_float('nc', 0.1, 1.0),
        title_w=trial.suggest_float('title_w', 0.1, 2.0),
        body_w=trial.suggest_float('body_w', 0.1, 2.0)
    )

    tr.bm25_ranker = solution.BM25Ranker(config)

    scores = tr.calculate_scores(ver='local')

    ndcg = calculate_ndcgk(scores, true_rels, k)
    return ndcg


def optimize_parameters(data_dir: Path, n_trials: int = 100) -> solution.BM25RankerConfig:
    """Run Optuna optimization with progress bar"""

    true_rels = load_true_rels(data_dir / "vkmarco-docdev-qrels.tsv")

    tr = solution.TextRanking(
        data_dir,
        "vkmarco-docdev-qrels_sub.csv",
        "vkmarco-docdev-queries"
    )

    study = optuna.create_study(direction="maximize")

    with tqdm(total=n_trials, desc="Optimizing parameters") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'best_value': study.best_value})

        study.optimize(
            lambda trial: objective(trial, tr, true_rels),
            n_trials=n_trials,
            callbacks=[callback]
        )

    best_params = study.best_params
    best_config = solution.BM25RankerConfig(
        k1=best_params['k1'],
        b=best_params['b'],
        nc=best_params['nc'],
        title_w=best_params['title_w'],
        body_w=best_params['body_w']
    )

    print(f"\nBest parameters found:")
    print(f"k1: {best_config.k1:.3f}")
    print(f"b: {best_config.b:.3f}")
    print(f"nc: {best_config.nc:.3f}")
    print(f"title_w: {best_config.title_w:.3f}")
    print(f"body_w: {best_config.body_w:.3f}")
    print(f"Best NDCG@10: {study.best_value:.3f}")

    return best_config


def main():
    best_config = optimize_parameters(DATA_DIR, n_trials=100)

    tr = solution.TextRanking(
        DATA_DIR,
        "vkmarco-docdev-qrels_sub.csv",
        "vkmarco-docdev-queries",
        ranker_config=best_config
    )

    scores = tr.calculate_scores(ver='local')

    true_rels = load_true_rels(DATA_DIR / "vkmarco-docdev-qrels.tsv")
    final_ndcg = calculate_ndcgk(scores, true_rels, 10)
    print(f"\nFinal NDCG@10: {final_ndcg:.3f}")


# distances + zones v0.1
#   0.5879126669881164
#   small dataset: 0.5877354782070126
# distances + zones v0.1
#   title_w: float = 0.3
#   body_w: float = 0.7
#   nc: 0.8
#   0.5883249648597936
#   nc: 0.9
#   0.5881310553819236
#   nc: 0.6
#   0.5885426164409874
#   nc: 0.4
#   0.588634721820811
#   k1: float = 0.25
#   b: float = 0.1
#   0.589
# bm25f (by claude)
#   w_title = 0.7  # title weight
#   w_body = 0.3  # body weight
#   b_title = 0.5  # title length normalization
#   b_body = 0.75  # body length normalization
#   k1 = 2.0
#   0.5820875936843558
# bm25f 2
#   w_title = 0.3  # title weight
#   w_body = 0.7  # body weight
#   0.5831006911497113
# bm25f * distances
#   0.5784615063806714
# bm25f + distances
#   0.5863463321267627
# bm25f + doc_distance
#   0.584627303485758
# bm25f + doc_distance (score * (0.5 + d_title * title_dist + d_body * doc_dist))
#   d_title = 0.7  # title nearness coefficient
#   d_body = 0.3  # body nearness coefficient
#   0.587097535192125

if __name__ == '__main__':
    main()
