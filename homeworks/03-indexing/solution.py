#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
import glob
import os
import pickle
import shutil
from itertools import chain
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import tqdm

from nltk import tokenize


def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def term2docs(term, index_path, term2file):
    index_file_ind = get_index_file_ind(term, term2file)
    if index_file_ind is None:
        return set()
    with open(index_path + f"/index_files/index_{index_file_ind}.pkl", 'rb') as f:
        reverse_index = pickle.load(f)
    docs = reverse_index.get(term, set())
    return docs


def get_index_file_ind(term, term2file):
    for t2f_fp in term2file:
        with open(t2f_fp, 'rb') as f:
            t2f_map = pickle.load(f)
        if term in t2f_map:
            return t2f_map[term]
    return None


def manage_filepath(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
    os.makedirs(filepath, exist_ok=True)


def save_term2file_maps(term2file, term2file_path, terms_count):
    manage_filepath(term2file_path)
    batch_size = terms_count // TERM2FILE_COUNT
    items = list(term2file.items())
    term2file_ind = 1
    for i in tqdm.tqdm(range(0, terms_count, batch_size),
                       total=TERM2FILE_COUNT,
                       desc="Saving term2file maps"):
        batch_term2file = dict(items[i:i + batch_size])
        with open(term2file_path + f"/term2file_{term2file_ind}.pkl", 'wb') as f:
            pickle.dump(batch_term2file, f)
        term2file_ind += 1


def save_reverse_index(index_path, reverse_index, term2file, terms_count):
    manage_filepath(index_path)
    terms = list(reverse_index.keys())
    file_ind = 1
    for i in tqdm.tqdm(range(0, terms_count, REVERSE_INDEX_BS),
                       total=(terms_count + REVERSE_INDEX_BS - 1) // REVERSE_INDEX_BS,
                       desc="Saving reverse index via splitted files"):
        batch_terms = terms[i:i + REVERSE_INDEX_BS]
        batch_index = {term: reverse_index[term] for term in batch_terms}
        for t in batch_index.keys():
            term2file[t] = file_ind
        with open(index_path + f"/index_{file_ind}.pkl", 'wb') as f:
            pickle.dump(batch_index, f)
        file_ind += 1


def process_query(query, index_path, term2file, total_docs):
    docs = term2docs(query[0], index_path, term2file)
    for term in query[1:]:
        docs = docs.intersection(term2docs(term, index_path, term2file))
    res = pd.DataFrame(np.zeros((total_docs, 2)), columns=['ObjectId', 'Label'])
    return docs, res


def update_res(res, docs, object_batch):
    res.iloc[:, 0] = object_batch.iloc[:, 0]
    mask = object_batch['DocumentId'].isin(docs)
    res.iloc[mask, 1] = 1
    return res.map(int)


def save_results(res, submission_file, first_batch):
    res.to_csv(submission_file, mode='w' if first_batch else 'a', header=first_batch, index=False)


DOCS_COUNT = 9025
QUERIES_COUNT = 100
REVERSE_INDEX_BS = 10
TERM2FILE_COUNT = 50


def main():
    # Парсим опции командной строки
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    # Какой у нас режим: построения индекса или генерации сабмишна?
    if args.build_index:
        # Тут вы должны:
        # - загрузить тексты документов из файла args.data_dir/vkmarco-docs.tsv
        # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
        # - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
        with open(args.data_dir + '/vkmarco-docs.tsv', 'rb') as f:
            queries_df = pd.read_csv(f, sep='\t', names=['doc_id', 'url', 'doc_title', 'doc_text'])
        doc_texts = (queries_df['doc_title'].fillna('') + ' ' + queries_df['doc_text'].fillna('')).to_numpy()
        doc_ids = queries_df['doc_id'].to_numpy()
        doc_terms = [preprocess(t) for t in doc_texts]
        reverse_index = {t: set() for t in chain.from_iterable(doc_terms)}
        for i in range(len(doc_ids)):
            for t in doc_terms[i]:
                reverse_index[t].add(doc_ids[i])

        term2file = {}
        terms_count = len(reverse_index)
        index_path = args.index_dir + '/index_files'
        save_reverse_index(index_path, reverse_index, term2file, terms_count)

        term2file_path = args.index_dir + f"/term2file"
        save_term2file_maps(term2file, term2file_path, terms_count)
    else:
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        #
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv

        queries_df = pd.read_csv(args.data_dir + '/vkmarco-doceval-queries.tsv', sep='\t',
                                 names=['query_id', 'query_text'],
                                 chunksize=REVERSE_INDEX_BS)
        index_path = args.index_dir
        term2file = glob.glob(index_path + '/term2file/*.pkl')
        objects = pd.read_csv(args.data_dir + "/objects.csv", chunksize=DOCS_COUNT)

        pbar = tqdm.tqdm(total=QUERIES_COUNT, desc="Proceeding queries")
        first_batch = True

        for ind, q_batch in enumerate(queries_df):
            for i in range(len(q_batch)):
                query = preprocess(q_batch.iloc[i]['query_text'])
                docs, res = process_query(query, index_path, term2file, DOCS_COUNT)
                objs = next(objects)
                res = update_res(res, docs, objs)
                save_results(res, args.submission_file, first_batch)
                first_batch = False
                del res
                pbar.update()
    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
