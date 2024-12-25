#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
import pickle
from itertools import chain
from timeit import default_timer as timer
import pandas as pd

from nltk import tokenize


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
        # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
        # - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
        with open(args.data_dir + '/vkmarco-docs.tsv', 'rb') as f:
            data = pd.read_csv(f, sep='\t', names=['doc_id', 'url', 'doc_title', 'doc_text'])
        doc_texts = (data['doc_title'].fillna('') + ' ' + data['doc_text']).tolist()
        doc_ids = data['doc_id'].tolist()
        doc_terms = [preprocess(t) for t in doc_texts]
        reverse_index = {t: set() for t in chain.from_iterable(doc_terms)}
        for i in range(len(doc_ids)):
            for t in doc_terms[i]:
                reverse_index[t].add(doc_ids[i])
        reverse_index['doc_ids'] = doc_ids
        with open(args.index_dir + '/index.pkl', 'wb') as f:
            pickle.dump(reverse_index, f)

    else:
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        #
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv
        with open(args.data_dir + '/vkmarco-doceval-queries.tsv', 'rb') as f:
            data = pd.read_csv(f, sep='\t', names=['query_id', 'query_text'])
        with open(args.index_dir + '/index.pkl', 'rb') as f:
            reverse_index = pickle.load(f)
        with open(args.data_dir + '/objects.csv', 'rb') as f:
            objects = pd.read_csv(f)

        # Сохраним информацию об объектах, чтоб потом не тыкать датафрейм
        object_dict = {}
        for _, row in objects.iterrows():
            key = (row['QueryId'], row['DocumentId'])
            object_dict[key] = row['ObjectId']
        all_object_ids = objects['ObjectId'].unique()

        submission_data = {}
        queries = data['query_text'].tolist()
        query_ids = data['query_id'].tolist()
        query_terms = [preprocess(t) for t in queries]

        for i in range(len(query_ids)):
            # для каждого запроса будем пересекать значения для терминов,
            # если для какого-то сет пуст - пуст и для запроса
            if query_terms[i][0] not in reverse_index:
                inter = set()
            else:
                inter = reverse_index[query_terms[i][0]]
                for j in range(1, len(query_terms[i])):
                    term = query_terms[i][j]
                    if term in reverse_index:
                        inter = inter.intersection(reverse_index[term])
                    else:
                        inter = set()
                        break
            for doc in inter:
                key = (query_ids[i], doc)
                if key in object_dict:
                    obj_id = object_dict[key]
                    submission_data[obj_id] = 1
        for obj_id in all_object_ids:
            if obj_id not in submission_data:
                submission_data[obj_id] = 0
        submission_df = pd.DataFrame(list(submission_data.items()), columns=['ObjectId', 'Label'])
        submission_df.to_csv(args.submission_file, index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()