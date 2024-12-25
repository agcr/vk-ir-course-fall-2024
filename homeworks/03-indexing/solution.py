#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
import os
import csv
import pickle
from timeit import default_timer as timer

from nltk import tokenize

def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def build_inverted_index(docs_tsv_path: str, index_dir: str):
    inverted_index = {}

    with open(docs_tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            doc_id = row[0]
            title = row[2]
            body = row[3]

            text = title + ' ' + body
            tokens = preprocess(text)

            for token in set(tokens):
                if token not in inverted_index:
                    inverted_index[token] = set()
                inverted_index[token].add(doc_id)

            if (i + 1) % 100000 == 0:
                print(f"Indexed {i+1} documents")

    # Сохраняем индекс в index_dir
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(inverted_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Index built! Total terms = {len(inverted_index):,}")


def load_inverted_index(index_dir: str):
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    with open(index_path, 'rb') as f:
        inverted_index = pickle.load(f)
    return inverted_index


def generate_submission(
    index_dir: str,
    data_dir: str,
    submission_file: str
):
    inverted_index = load_inverted_index(index_dir)

    queries_path = os.path.join(data_dir, 'vkmarco-doceval-queries.tsv')
    queries_dict = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            query_id = row[0]
            query_text = row[1]
            query_tokens = preprocess(query_text)
            queries_dict[query_id] = query_tokens

    objects_path = os.path.join(data_dir, 'objects.csv')
    sample_sub_path = os.path.join(data_dir, 'sample_submission.csv')

    with open(sample_sub_path, 'r', encoding='utf-8') as sf, \
         open(submission_file, 'w', encoding='utf-8', newline='') as outf, \
         open(objects_path, 'r', encoding='utf-8') as objf:

        sub_reader = csv.DictReader(sf)
        fieldnames = sub_reader.fieldnames  # ['ObjectId', 'Label']
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()

        obj_reader = csv.DictReader(objf)
        object_map = {}
        for row_obj in obj_reader:
            object_id = row_obj['ObjectId']
            query_id = row_obj['QueryId']
            doc_id = row_obj['DocumentId']
            object_map[object_id] = (query_id, doc_id)

        sf.seek(0)
        next(sf)
        for line in sf:
            object_id_str, _ = line.strip().split(',')
            q_id, d_id = object_map[object_id_str]

            query_terms = queries_dict[q_id]

            if not query_terms:
                label = 1
            else:
                label = 1
                for term in query_terms:
                    postings = inverted_index.get(term)
                    if (not postings) or (d_id not in postings):
                        label = 0
                        break

            writer.writerow({
                'ObjectId': object_id_str,
                'Label': label
            })


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

    docs_tsv = os.path.join(args.data_dir, 'vkmarco-docs.tsv')
    if args.build_index:
        build_inverted_index(docs_tsv, args.index_dir)
    else:
        if not args.submission_file:
            raise ValueError("Please specify --submission_file in generation mode")
        generate_submission(args.index_dir, args.data_dir, args.submission_file)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
