#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import os
import argparse
from timeit import default_timer as timer
from tqdm import tqdm

from nltk import tokenize

def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]

def search_by_index(query_tokens, index):
    try:
        doc_ids = set(index.get(query_tokens[0], set()))
        for i in range(1, len(query_tokens)):
            doc_ids = doc_ids.intersection(index.get(query_tokens[i], set()))
        return list(doc_ids)
    except KeyError as e:
        return []


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
        index = {}
        with open(os.path.join(args.data_dir, 'vkmarco-docs.tsv'), 'r', encoding='utf-8') as file:
            for _, line in tqdm(enumerate(file), desc="Create index"):
                document = line.strip().split('\t')
                doc_id = document[0]
                # doc_ref = document[1]
                doc_title = document[2]
                doc_body = document[3]

                tokens = preprocess(doc_title + ' ' + doc_body)

                for token in tokens:
                    index[token] = index.get(token, set())
                    index[token].update([doc_id])

        with open(os.path.join(args.index_dir, 'index.csv'), 'w', encoding='utf-8') as file:
            for token, doc_ids in tqdm(index.items(), desc="Save index"):
                sorted_doc_ids = sorted(doc_ids)
                str_doc_ids = '\t'.join(sorted_doc_ids)
                file.write(f'{token}\t{str_doc_ids}\n')

    else:
        index = {}
        with open(os.path.join(args.index_dir, 'index.csv'), 'r', encoding='utf-8') as file:
            for _, line in tqdm(enumerate(file), desc="Load index"):
                idx = line.strip().split('\t')
                index[idx[0]] = idx[1:]

        prev_query_id = None
        with open(os.path.join(args.data_dir, 'vkmarco-doceval-queries.tsv'), 'r', encoding='utf-8') as file_query:
            with open(os.path.join(args.data_dir, 'objects.csv'), 'r', encoding='utf-8') as file_objects:
                _ = file_objects.readline()
                with open(args.submission_file, 'w', encoding='utf-8') as file_submission:
                    file_submission.write('ObjectId,Label\n')
                    for line in tqdm(file_objects, desc="Search by index"):
                        objct = line.strip().split(',')
                        obj_id, query_id, doc_id = objct

                        if query_id != prev_query_id:
                            query_text = file_query.readline().strip().split('\t')[1]
                            query_tokens = preprocess(query_text)

                            doc_ids = search_by_index(query_tokens, index)
                            prev_query_id = query_id

                        if doc_id in doc_ids:
                            file_submission.write(f"{obj_id},1\n")
                        else:
                            file_submission.write(f"{obj_id},0\n")

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
