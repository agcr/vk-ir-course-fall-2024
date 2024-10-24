#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
from timeit import default_timer as timer
import os
import tqdm

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
        indexes = {}
        with open(os.path.join(args.data_dir, 'vkmarco-docs.tsv'), 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_values = line.strip().split('\t')
                doc_id = line_values[0]

                title = line_values[2]
                body = line_values[3]

                tokens = preprocess(title + " " + body)

                for token in tokens:
                    indexes[token] = indexes.get(token, set())
                    indexes[token].update([doc_id])
        # saving
        with open(os.path.join(args.index_dir, 'index.txt'), 'w', encoding='utf-8') as f:
            for token, index_ids in indexes.items():
                sorted_idxs = sorted(index_ids)
                docs_str = "\t".join(sorted_idxs)
                f.write(f'{token}\t{docs_str}\n')

    else:
        # load indexes
        indexes = {}
        with open(os.path.join(args.index_dir, 'index.txt'),
                  'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_values = line.strip().split('\t')
                token = line_values[0]
                indexes[token] = line_values[1:]

        prev_query_id = None
        with open(os.path.join(args.data_dir, 'vkmarco-doceval-queries.tsv'),
             'r', encoding='utf-8') as f_query_txt:

            with open(os.path.join(args.data_dir, 'objects.csv'),
                 'r', encoding='utf-8') as f_objects:
                _ = f_objects.readline()  # skip header

                with open(args.submission_file, 'w', encoding='utf-8') as f_subm:
                    f_subm.write("ObjectId,Label\n")

                    for line in tqdm.tqdm(f_objects, desc="Iteration through objects"):
                        line_values = line.strip().split(',')
                        object_id, query_id, document_id = line_values
                        # document_id = document_id.strip()
                        if query_id != prev_query_id:
                            query_text = f_query_txt.readline().strip().split('\t')[1]
                            query_tokens = preprocess(query_text)

                            docs_ids = search_in_indexes(query_tokens, indexes)
                            prev_query_id = query_id
                        if document_id in docs_ids:
                            label = 1
                        else:
                            label = 0
                        line = f"{object_id},{label}\n"
                        f_subm.write(line)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


def load_query_text(search_query_id, data_dir):
    with open(os.path.join(data_dir, 'vkmarco-doceval-queries.tsv'),
              'r', encoding='utf-8') as f:
        # print(f"{search_query_id=}")
        for i, line in enumerate(f):
            line_values = line.split('\t')
            query_id = line_values[0]
            if query_id == search_query_id:
                query_text = line_values[1]
                return query_text
            else:
                continue
    raise ValueError("Query id is not found")


def search_in_indexes(query: list, indexes: dict) -> list:
    """
    Searches query tokens in inversed indexes, then intersects them.
    Returns founded docs ids and bool flag that shows if all tokens
    input tokens were indexed.
    """

    try:
        result = indexes[query[0]]

        res_len = len(result)
        for i in range(1, len(query)):
            result, res_len = intersect_sorted_(result, indexes[query[i]])
            del result[res_len:]
            # чтобы было оптимально по памяти результат пересечения сразу записывается
            # в переменную result, а не в новую переменную
        return sorted(result)

    except KeyError as e:
        # print("Cant found token:", e)
        return []


def intersect_sorted_(arr1, arr2):
    """
    Intersects two sorted arrays.
    Result is placed in arr1.

    Returns arr1 and  len of arr1.
    """
    i = j = k = 0
    # result = []
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            arr1[k] = arr1[i]
            k += 1
            # result.append(arr1[i])
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1
    return arr1, k


if __name__ == "__main__":
    main()
#  python .\solution.py --submission_file=result/sub.csv --build_index --index_dir=indexes data/
#  python .\solution.py --submission_file=result/sub.csv  --index_dir=indexes data/
