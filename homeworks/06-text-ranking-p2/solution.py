#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Text ranking homework solution"""

import argparse
import gc
import os.path
import pickle
import time
from timeit import default_timer as timer
import numpy as np
from catboost import utils
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pymorphy3 as pymorphy
from collections import defaultdict
from tqdm.auto import tqdm
from pandarallel import pandarallel
import optuna
from statistics import mean

pandarallel.initialize(progress_bar=True)

"""
Параметры были выучены с помощью библиотеки optuna (изначально PARAMS был пустым)
В целях воспроизводимости submission'а вынес их в словарь, но код их получения оставил
"""
PARAMS = {'w_t': 0.9688644739512802, 'b_t': 0.7047412598104205,
          'w_b': 0.10246279491683703, 'b_b': 0.13969069421098465,
          'k_1': 0.25651770779081284, 'b': 0.10015744025550016}


def need_calc(files_list):
    return not all(map(os.path.exists, files_list))


def get_best_normal_form(morph, word):
    """Helper function: applies analyzer to word and extracts best normal form"""
    parsed = morph.parse(word)
    if not parsed:
        print(f"Can't parse {word}")
        return word
    best_form = parsed[0]
    return best_form.normal_form


class MorphCountVectorizer(CountVectorizer):
    def __init__(self, lang='ru', **kwargs):
        super().__init__(**kwargs)
        self.lang = lang

    def build_analyzer(self):
        """Called only once so we don't need to cache analyzer"""
        analyzer = super().build_analyzer()
        morph = pymorphy.MorphAnalyzer(lang=self.lang)
        return lambda text: (get_best_normal_form(morph, word) for word in analyzer(text))


def concat_dfs(dfs_title, dfs_body):
    dfs = defaultdict(int, {k: dfs_body[k] for k in set(dfs_title) & set(dfs_body)})
    dfs |= {k: dfs_body.get(k, 0) + dfs_title.get(k, 0) for k in set(dfs_title) ^ set(dfs_body)}
    return dfs


def predict_to_subm(predicts):
    res = {}
    for query_id in predicts:
        bm25_scores = []
        cur_query_docs = list(predicts[query_id].keys())
        for doc_id in cur_query_docs:
            bm25_scores.append(predicts[query_id][doc_id])
        inds = np.argsort(np.array(bm25_scores, dtype=np.float64))[::-1]
        res[query_id] = [cur_query_docs[ind] for ind in inds]
    return res


def func_to_row(analyzer, row_to_apply):
    from collections import Counter
    lexemes_title = tuple(analyzer(row_to_apply[2]))
    lexemes_body = tuple(analyzer(row_to_apply[3]))
    c_title = Counter(lexemes_title)
    c_body = Counter(lexemes_body)
    return c_title, c_body


def bm25_calc(query_docs, query_lemmes, dfs, tfs, docs_l, mean_l, all_docs,
              k, b):
    ans = defaultdict(dict)
    for query_id in query_docs:
        assert all([(all_docs - dfs[l] + 0.5) > 0 for l in query_lemmes[query_id]])
        idfs = {l: np.log((all_docs - dfs[l] + 0.5) / (dfs[l] + 0.5)) for l in query_lemmes[query_id]}
        cur_query_docs = list(query_docs[query_id].keys())
        for d in cur_query_docs:
            bm25 = 0
            for l in query_lemmes[query_id]:
                tf = tfs[l].get(d, 0)
                bm25 += (idfs[l] * tf * (k + 1) /
                         (tf + k * (1 - b + b * docs_l.get(d, 0) / mean_l)))
            ans[query_id][d] = bm25
    return ans


def make_bm25f_subm(doc_filename, analyzer, query_docs, query_lemmes, doc_query):
    need_new = need_calc(["consts_title.pkl", "consts_body.pkl",
                          "tfs_title.pkl", "tfs_body.pkl",
                          "dfs_title.pkl", "dfs_body.pkl",
                          "all_docs.pkl"])
    if need_new:
        save_docs_data(analyzer, doc_query, query_lemmes,
                       doc_filename,
                       'tfs_title.pkl', 'tfs_body.pkl',
                       'dfs_title.pkl', 'dfs_body.pkl',
                       'consts_title.pkl', 'consts_body.pkl',
                       'all_docs.pkl')
    with open('tfs_title.pkl', 'rb') as f:
        tfs_title = pickle.load(f)
    with open('tfs_body.pkl', 'rb') as f:
        tfs_body = pickle.load(f)
    with open('dfs_title.pkl', 'rb') as f:
        dfs_title = pickle.load(f)
    with open('dfs_body.pkl', 'rb') as f:
        dfs_body = pickle.load(f)
    with open('consts_title.pkl', 'rb') as f:
        docs_l_t, mean_t = pickle.load(f)
    with open('consts_body.pkl', 'rb') as f:
        docs_l_b, mean_b = pickle.load(f)
    with open('all_docs.pkl', 'rb') as f:
        all_docs = pickle.load(f)
    dfs = concat_dfs(dfs_title, dfs_body)
    lengths = {k: docs_l_t[k] + docs_l_b[k] for k in docs_l_t}
    mean_l = mean(lengths.values())
    tws = defaultdict(dict)
    for l in tfs_title:
        for d in tfs_title[l]:
            tws[l][d] = (PARAMS['w_t'] * tfs_title[l][d] / (
                    1 - PARAMS['b_t'] + PARAMS['b_t'] * docs_l_t[d] / mean_t) +
                         PARAMS['w_b'] * tfs_body[l][d] / (
                                 1 - PARAMS['b_b'] + PARAMS['b_b'] * docs_l_b[d] / mean_b))
    bm25_zonned_scores = bm25_calc(query_docs=query_docs, query_lemmes=query_lemmes,
                                   dfs=dfs, tfs=tws, docs_l=lengths, mean_l=mean_l, all_docs=all_docs,
                                   k=PARAMS['k_1'], b=PARAMS['b'])
    return predict_to_subm(bm25_zonned_scores)


def select_texts(doc_filename, docs_collection, output_name):
    dataframes = []
    c = 0
    with pd.read_csv(doc_filename, delimiter='\t',
                     chunksize=10_000, header=None) as reader:
        for chunk in reader:
            print("Chunk")
            df_to_add = chunk[chunk[0].isin(docs_collection)]
            c += df_to_add.shape[0]
            dataframes.append(df_to_add)
    print(c)
    new_df = pd.concat(dataframes, ignore_index=True)
    new_df.to_csv(output_name, sep='\t', index=False)


def save_docs_data(analyzer, train_doc_query_mapping, train_query_lemmes,
                   doc_filename,
                   tfs_title_filename, tfs_body_filename,
                   dfs_title_filename, dfs_body_filename,
                   consts_title_filename, consts_body_filename,
                   all_docs_filename):
    c_ind = 0
    all_docs = 0
    tfs_title = defaultdict(dict)
    tfs_body = defaultdict(dict)
    dfs_title = defaultdict(int)
    dfs_body = defaultdict(int)
    docs_l_t = {}
    docs_l_b = {}
    lengths_t = []
    lengths_b = []
    with pd.read_csv(doc_filename, delimiter='\t',
                     chunksize=10_000, keep_default_na=False) as reader:
        for chunk in reader:
            c_ind += 1
            print(f"Chunk {c_ind}")
            chunk[["counts_title", "counts_body"]] = chunk.parallel_apply(lambda r: func_to_row(analyzer, r),
                                                                          axis=1, result_type='expand')
            for ind, row in chunk.iterrows():
                all_docs += 1
                doc_id = row[0]
                c_t = row["counts_title"]
                c_b = row["counts_body"]
                for l in c_t:
                    dfs_title[l] += 1
                for l in c_b:
                    dfs_body[l] += 1
                lex_length_t = sum(c_t.values())
                lex_length_b = sum(c_b.values())
                lengths_t.append(lex_length_t)
                lengths_b.append(lex_length_b)
                docs_l_t[doc_id] = lex_length_t
                docs_l_b[doc_id] = lex_length_b
                for query_id in train_doc_query_mapping[doc_id]:
                    query_lms = train_query_lemmes[query_id]
                    for ql in query_lms:
                        tfs_title[ql][doc_id] = c_t.get(ql, 0)
                        tfs_body[ql][doc_id] = c_b.get(ql, 0)
            if c_ind % 3 == 0:
                time.sleep(60.5)
        mean_t = mean(lengths_t)
        mean_b = mean(lengths_b)
        with open(tfs_title_filename, 'wb') as f:
            pickle.dump(tfs_title, f)
        with open(tfs_body_filename, 'wb') as f:
            pickle.dump(tfs_body, f)
        with open(dfs_title_filename, 'wb') as f:
            pickle.dump(dfs_title, f)
        with open(dfs_body_filename, 'wb') as f:
            pickle.dump(dfs_body, f)
        with open(consts_title_filename, 'wb') as f:
            pickle.dump((docs_l_t, mean_t), f)
        with open(consts_body_filename, 'wb') as f:
            pickle.dump((docs_l_b, mean_b), f)
        with open(all_docs_filename, 'wb') as f:
            pickle.dump(all_docs, f)


def calc_ndcg(mapping_predict, mapping_qrels):
    rels = []
    scores = []
    groups = []
    for query_id in mapping_predict:
        doc_ids = mapping_predict[query_id]
        doc_amount = len(doc_ids)
        for i in range(doc_amount):
            rels.append(mapping_qrels[query_id][doc_ids[i]])
            scores.append(doc_amount - i)
            groups.append(query_id)
    return utils.eval_metric(rels, scores, 'NDCG:top=10;type=Exp', group_id=groups)[0]


def wrap_bm25f_checker(query_docs,
                       query_lemmes,
                       tfs_title, tfs_body,
                       dfs,
                       docs_l_title, docs_l_body,
                       docs_l,
                       mean_l_title, mean_l_body,
                       mean_l,
                       all_docs, query_doc_qrel):
    def params_checker(trial):
        w_t = trial.suggest_float("w_t", 0.1, 2)
        b_t = trial.suggest_float("b_t", 0.1, 2)
        w_b = trial.suggest_float("w_b", 0.1, 2)
        b_b = trial.suggest_float("b_b", 0.1, 2)
        k_1 = trial.suggest_float("k_1", 0.1, 2)
        b = trial.suggest_float("b", 0.1, 2)
        tws = defaultdict(dict)
        for l in tfs_title:
            for d in tfs_title[l]:
                tws[l][d] = (w_t * tfs_title[l][d] / (1 - b_t + b_t * docs_l_title[d] / mean_l_title) +
                             w_b * tfs_body[l][d] / (1 - b_b + b_b * docs_l_body[d] / mean_l_body))
        bm25_zonned_scores = bm25_calc(query_docs=query_docs, query_lemmes=query_lemmes,
                                       dfs=dfs, tfs=tws, docs_l=docs_l, mean_l=mean_l, all_docs=all_docs,
                                       k=k_1, b=b)
        res = predict_to_subm(bm25_zonned_scores)
        return calc_ndcg(res, query_doc_qrel)

    return params_checker


def train_bm25f(analyzer, data_path):
    df = pd.read_csv(os.path.join(data_path, "vkmarco-docdev-qrels.tsv"), delimiter=" ",
                     header=None)
    train_query_docs_mapping = defaultdict(dict)
    train_doc_query_mapping = defaultdict(list)
    for ind, row in df.iterrows():
        train_query_docs_mapping[row[0]][row[2]] = row[3]
        train_doc_query_mapping[row[2]].append(row[0])
    del df
    gc.collect()
    train_query_lemmes = {}
    df = pd.read_csv(os.path.join(data_path, "vkmarco-docdev-queries.tsv"), delimiter="\t", header=None)
    for ind, row in df.iterrows():
        analyzed = tuple(analyzer(row[1]))
        train_query_lemmes[row[0]] = analyzed
    del df
    gc.collect()
    if need_calc(['train_texts.csv']):
        select_texts(os.path.join(data_path, "vkmarco-docs.tsv"),
                     train_doc_query_mapping.keys(), "train_texts.csv")
    need = need_calc(['tfs_train_title.pkl', 'tfs_train_body.pkl', 'dfs_train_title.pkl',
                      'dfs_train_body.pkl', 'consts_train_title.pkl', 'consts_train_body.pkl',
                      'all_train_docs.pkl'])
    if need:
        save_docs_data(analyzer, train_doc_query_mapping, train_query_lemmes,
                       'train_texts.csv',
                       'tfs_train_title.pkl', 'tfs_train_body.pkl',
                       'dfs_train_title.pkl', 'dfs_train_body.pkl',
                       'consts_train_title.pkl', 'consts_train_body.pkl',
                       'all_train_docs.pkl')
    with open('tfs_train_title.pkl', 'rb') as f:
        tfs_title = pickle.load(f)
    with open('tfs_train_body.pkl', 'rb') as f:
        tfs_body = pickle.load(f)
    with open('dfs_train_title.pkl', 'rb') as f:
        dfs_title = pickle.load(f)
    with open('dfs_train_body.pkl', 'rb') as f:
        dfs_body = pickle.load(f)
    with open('consts_train_title.pkl', 'rb') as f:
        docs_l_t, mean_t = pickle.load(f)
    with open('consts_train_body.pkl', 'rb') as f:
        docs_l_b, mean_b = pickle.load(f)
    with open('all_train_docs.pkl', 'rb') as f:
        all_docs = pickle.load(f)
    dfs = concat_dfs(dfs_title, dfs_body)
    lengths = {k: docs_l_t[k] + docs_l_b[k] for k in docs_l_t}
    mean_l = mean(lengths.values())
    optuna_func = wrap_bm25f_checker(query_docs=train_query_docs_mapping,
                                     query_lemmes=train_query_lemmes,
                                     tfs_title=tfs_title, tfs_body=tfs_body,
                                     dfs=dfs,
                                     docs_l_title=docs_l_t, docs_l_body=docs_l_b,
                                     docs_l=lengths,
                                     mean_l_title=mean_t, mean_l_body=mean_b,
                                     mean_l=mean_l,
                                     all_docs=all_docs, query_doc_qrel=train_query_docs_mapping)
    study_params = optuna.create_study(direction="maximize")
    n_trials = 500
    with tqdm(total=n_trials, desc="Optimization") as pbar:
        def callback(*_):
            pbar.update(1)

        study_params.optimize(optuna_func, n_trials=n_trials, callbacks=[callback])
    print(study_params.best_params)
    print(study_params.best_value)
    with open('trained_constants.pkl', 'wb') as f:
        pickle.dump(study_params.best_params, f)


def main():
    global PARAMS
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Text ranking homework solution')
    parser.add_argument('--submission_file', required=True, help='output Kaggle submission file')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()
    start = timer()
    analyze = MorphCountVectorizer().build_analyzer()
    df = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"), delimiter=",")
    query_docs_mapping = defaultdict(dict)
    doc_query_mapping = defaultdict(list)
    for ind, row in df.iterrows():
        query_docs_mapping[row['QueryId']][row['DocumentId']] = 0
        doc_query_mapping[row['DocumentId']].append(row['QueryId'])
    del df
    gc.collect()
    query_lemmes_mapping = {}
    df = pd.read_csv(os.path.join(args.data_dir, "vkmarco-doceval-queries.tsv"), delimiter="\t", header=None)
    for ind, row in df.iterrows():
        analyzed = tuple(analyze(row[1]))
        query_lemmes_mapping[row[0]] = analyzed
    del df
    gc.collect()
    if need_calc(["useful_texts.csv"]):
        select_texts(os.path.join(args.data_dir, "vkmarco-docs.tsv"), doc_query_mapping.keys(),
                     "useful_texts.csv")
    """Из-за случайности optuna и сложности воспроизведения результатов, 
    вынес обученные параметры в словарь. 
    При решении задачи строки ниже были раскомментированы, а PARAMS был пустым"""
    # if not os.path.exists("trained_constants.pkl"):
    #     train_bm25f(analyze, args.data_dir)
    #     with open('trained_constants.pkl', 'rb') as f:
    #         PARAMS = pickle.load(f)
    print("Making submission...")
    ans = make_bm25f_subm("useful_texts.csv", analyze,
                          query_docs_mapping, query_lemmes_mapping, doc_query_mapping)
    with open(args.submission_file, 'w') as sub_f:
        sub_f.write("QueryId,DocumentId\n")
        for q_id in ans:
            for doc in ans[q_id]:
                sub_f.write(f"{q_id},{doc}\n")

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
