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
        filename = os.path.join(args.data_dir, "vkmarco-docs.tsv")
        df = pd.read_csv(filename, sep='\t', names=['id', 'url', 'title', 'body'])
        df['id'] = df['id'].apply(lambda x: int(x[1:]))
        df['text'] = df['title'] + " " + df['body']
        df.drop(['title', 'body', 'url'], axis=1, inplace=True)
        df = df.set_index('id')
        df.dropna(inplace=True)
        dictionary = defaultdict(list)
        for index, row in df.iterrows():
            # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
            for word in preprocess(row.text):
                dictionary[word].append(index)
        dictionary = dict(dictionary)
        for key in dictionary.keys():
            dictionary[key].sort()
        # - сохранить получивший обратный индекс в папку, переданную через параметр args.index_dir
        with open(os.path.join(args.index_dir, "index"), 'w') as out:
            out.write(str(len(dictionary)))
            for key in dictionary.keys():
                out.write(key)
            for key in dictionary.keys():
                out.write(' '.join(map(str, dictionary[key])))
    else:
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        with open(os.path.join(args.index_dir, "index"), 'r') as index_file:
            pass
        labels = dict()
        with open(os.path.join(args.data_dir, "vkmarco-doceval-queries.tsv"), 'r') as infile:
            line = infile.readline()
            id, query = line.split(' ', 1)
            id = int(id)


        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        #
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv
        pass

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
