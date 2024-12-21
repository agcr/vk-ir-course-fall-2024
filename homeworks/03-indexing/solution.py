#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
import os

import pandas as pd
from timeit import default_timer as timer
from collections import defaultdict

from nltk import tokenize

def intersect(a, b):
    res = []
    pa = 0
    pb = 0
    while pa < len(a) and pb < len(b):
        if a[pa] == b[pb]:
            res.append(a[pa])
            pa += 1
            pb += 1
        elif a[pa] < b[pb]:
            pa += 1
        else:
            pb += 1
    return res

def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]

def process_query(query, args):
    # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
    mapped_docs = None
    with open(os.path.join(args.index_dir, INDEX_FILE_NAME), 'r') as index_file:
        for line in index_file:
            line_split = line.split(',', 1)
            if line_split[0] in query:
                if mapped_docs is None:
                    mapped_docs = list(map(int, line_split[1].split()))
                else:
                    current_docs = list(map(int, line_split[1].split()))
                    mapped_docs = intersect(mapped_docs, current_docs)
                    del current_docs
            del line_split
    if mapped_docs is None:
        return []
    return mapped_docs

INDEX_FILE_NAME = "index_file.txt"

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
        print('Building index')
        # Тут вы должны:
        # - загрузить тексты документов из файла args.data_dir/vkmarco-docs.tsv
        filename = os.path.join(args.data_dir, "vkmarco-docs.tsv")
        df = pd.read_csv(filename, sep='\t', names=['id', 'url', 'title', 'body'])
        df['id'] = df['id'].apply(lambda x: int(x[1:]))
        df['text'] = df['title'] + " " + df['body']
        df.drop(['title', 'body', 'url'], axis=1, inplace=True)
        df = df.set_index('id')
        df.fillna("", inplace=True)
        dictionary = defaultdict(list)
        for index, row in df.iterrows():
            # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
            for word in preprocess(row.text):
                dictionary[word].append(index)
        dictionary = dict(dictionary)
        for key in dictionary.keys():
            dictionary[key].sort()
        # - сохранить получивший обратный индекс в папку, переданную через параметр args.index_dir
        if not os.path.exists(args.index_dir):
            os.makedirs(args.index_dir)
        with open(os.path.join(args.index_dir, INDEX_FILE_NAME), 'w') as out:
            for key in dictionary.keys():
                out.write(key + ',' + ' '.join(map(str, dictionary[key])) + '\n')
        print('Indexing complete')
    else:
        last_query_idx = -1
        last_query_docs = []
        with open(args.submission_file, 'w') as submission:
            submission.write('ObjectId,Label\n')
            with open(os.path.join(args.data_dir, "objects.csv"), 'r') as objects:
                objects.readline()
                for line in objects:
                    ObjectId, QueryId, DocumentId = line.split(',')
                    QueryId = int(QueryId)
                    if QueryId != last_query_idx:
                        last_query_idx = QueryId
                        del last_query_docs
                        print(last_query_idx)
                        with open(os.path.join(args.data_dir, "vkmarco-doceval-queries.tsv"), 'r') as infile:
                            for in_line in infile:
                                idx, text = in_line.split('\t', 1)
                                idx = int(idx)
                                if idx == last_query_idx:
                                    words = set(preprocess(text))
                                    break
                        last_query_docs = process_query(words, args)
                    is_relevant = int(DocumentId[1:]) in last_query_docs
                    submission.write(ObjectId + "," + ("1" if is_relevant else "0") + "\n")

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
