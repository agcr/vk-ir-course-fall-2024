#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python3 solution.py --build_index --index_dir=index boolean-retrieval-homework-vk-ir-fall-2024
python3 solution.py --submission_file=submission.csv --index_dir=index boolean-retrieval-homework-vk-ir-fall-2024
"""

"""Indexing homework solution"""

import argparse
from timeit import default_timer as timer
import json
import os 
import pandas as pd

from nltk import tokenize

def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]

def get_relevant_docs(words, index):
    relevant_docs = None
    for word in words:
        if word not in index:
            continue
        if relevant_docs is None:
            relevant_docs = set(index[word])
            continue
        relevant_docs = relevant_docs.intersection(set(index[word]))
    return relevant_docs

def is_relevant(row):
    global TOTAL_RELEVANT
    if (str(row['QueryId']), row['DocumentId']) in TOTAL_RELEVANT:
        return 1
    return 0

def main():
    global TOTAL_RELEVANT
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
        index = {}
        with open(args.data_dir + "/vkmarco-docs.tsv", "r") as pages:
            for line in pages:
                fields = line.rstrip('\n').split('\t')
                docid, url, title, body = fields
                total_text = preprocess(title + " " + body)
                for word in total_text:
                    if word not in index:
                        index[word] = [docid]
                    else:
                        index[word].append(docid)
        if not os.path.exists(args.index_dir):
            os.makedirs(args.index_dir)
        with open(args.index_dir + "/index.json", "w", encoding="utf-8") as file_index:
            json.dump(index, file_index)
        # Тут вы должны:
        # - загрузить тексты документов из файла args.data_dir/vkmarco-docs.tsv
        # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
        # - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
    else:
        index = {}
        with open(args.index_dir + "/index.json", "r", encoding="utf-8") as file_index:
            index = json.load(file_index)

        with open(args.data_dir + "/vkmarco-doceval-queries.tsv", "r") as querys:
            for line in querys:
                fields = line.rstrip('\n').split('\t')
                query_id, query_text = fields
                query_words = preprocess(query_text)
                relevant_docs = get_relevant_docs(query_words, index)
                print(relevant_docs, "\n")
                for doc in relevant_docs:
                    TOTAL_RELEVANT.append((query_id, doc))
        
        df = pd.read_csv(args.data_dir + "/objects.csv")
        df['Label'] = df.apply(is_relevant, axis=1)
        df.drop(columns=['QueryId', 'DocumentId'], axis=1, inplace=True)
        df.to_csv(args.data_dir + "/" + args.submission_file, index=False)
        
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        #
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
