#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
from timeit import default_timer as timer
from collections import Counter, defaultdict
import csv
import hashlib

from nltk import tokenize
import numpy as np
import pandas as pd

def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]

def hash_term(term):
    return int(hashlib.md5(term.encode()).hexdigest(), 16) % (10**8)

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
        inverted_index = defaultdict(set)
        # - загрузить тексты документов из файла args.data_dir/vkmarco-docs.tsv
        with open(f'{args.data_dir}/vkmarco-docs.tsv', 'r', encoding='utf-8') as file:
        # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
            reader = csv.reader(file, delimiter='\t')

            for i, text in enumerate(reader):
                tokens = preprocess(f'{text[2]}. {text[3]}')
                doc = text[0]
                
                for token in tokens:
                    hashed_token = hash_term(token)
                    inverted_index[hashed_token].add(doc)

        # - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
        with open(f'{args.index_dir}/inverted_index.txt', 'w', encoding='utf-8') as file:
            for word, docs_id in inverted_index.items():
                file.write(f'{word}: {sorted(docs_id)}\n')


    else:
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        with open(f'{args.data_dir}/vkmarco-doceval-queries.tsv', 'r', encoding='utf-8') as file:
            doceval_queries = file.readlines()
        
        with open(f'{args.index_dir}/inverted_index.txt', 'r', encoding='utf-8') as file:
            file_lines = file.readlines()

        terms = np.zeros(len(file_lines))
        docs_ids = []

        for i, line in enumerate(file_lines):
            word, doc_ids = line.split(':')
            doc_ids_cleaned = doc_ids.strip()[1:-1].replace(' ', '').replace('\'', '').split(',')
            docs_ids.append(list(doc_ids_cleaned))

            terms[i] = int(word)

        # Сортируем для дальнейшего быстрого бинарного поиска
        sorted_indices = np.argsort(terms)

        terms_sorted = terms[sorted_indices]
        docs_ids_sorted = [docs_ids[i] for i in sorted_indices]

        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        query_doc = {}

        with open('./datasets/objects.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Пропускаем заголовок
            for parts in reader:
                key = int(parts[0])
                value = parts[1:]
                query_doc[tuple(value)] = key
        
        num_docs = 9025

        result_matrix = np.zeros((len(doceval_queries) * num_docs, 2))
        result_matrix[:, 0] = np.arange(1, len(doceval_queries) * num_docs + 1)


        for i, query in enumerate(doceval_queries):
            docs_ids = []
            tokens = preprocess(query[1])
            num_query = query[0]

            for token in tokens:
                hashed_token = hash_term(token)
                index_token = np.searchsorted(terms_sorted, hashed_token)
                if index_token < len(terms_sorted) and terms_sorted[index_token] == hashed_token:
                    docs_ids.append(set(docs_ids_sorted[index_token]))

            if len(docs_ids) > 0:
                result = docs_ids[0]
                for doc_id in docs_ids[1:]:
                    result = result.intersection(doc_id)

                objects_id = [query_doc[(num_query, doc)] for doc in result]
                if len(objects_id) > 0:
                    result_matrix[np.array(objects_id) - 1, 1] = 1

        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv
        submission = pd.DataFrame(result_matrix, columns=["ObjectId", "Label"])
        submission["Label"] = submission["Label"].fillna(0).astype('int8')
        submission['ObjectId'] = submission['ObjectId'].astype('Int32')
        submission.to_csv(f'{args.submission_file}', index=False)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
