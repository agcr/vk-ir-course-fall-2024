#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""
import argparse
import json
import os
from codecs import open as copen
from collections import defaultdict
from hashlib import sha224
from pathlib import Path
from timeit import default_timer as timer

from nltk import tokenize


class SearchResults:
    def __init__(self):
        self.results = set()

    def add(self, query_id, document_indices):
        for document_index in document_indices:
            self.results.add(f"{query_id}, {document_index}")

    def save_submission(self, objects_file, submission_file):
        with copen(submission_file, 'w', encoding='utf-8') as submission_f:
            with copen(objects_file, 'r', encoding='utf-8') as objects_f:
                for line in objects_f:
                    if line.startswith('ObjectId'):
                        submission_f.write("ObjectId,Relevance\n")
                        continue
                    object_id, query_id, document_id = line.strip().split(',')
                    relevance = "1" if f"{query_id}, {document_id}" in self.results else "0"
                    submission_f.write(f"{object_id},{relevance}\n")


def preprocess(text):
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]


def save_index(index_dir, inverted_index):
    os.makedirs(index_dir, exist_ok=True)
    partitioned_index = defaultdict(dict)
    for term, postings in inverted_index.items():
        partition_key = int(sha224(term.encode("utf-8")).hexdigest(), 16) % 31
        partitioned_index[partition_key][term] = postings

    for key, postings in partitioned_index.items():
        file_path = os.path.join(index_dir, f"{key}.json")
        with open(file_path, 'w') as f:
            json.dump(postings, f, default=lambda obj: list(obj) if isinstance(obj, set) else obj)


def build_index(data_dir, index_dir):
    inverted_index = defaultdict(set)
    input_file = os.path.join(data_dir, 'vkmarco-docs.tsv')

    with copen(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            splitted_line = line.strip().split('\t')
            doc_id, text_header = splitted_line[0], splitted_line[3] + splitted_line[2]
            for word in preprocess(text_header):
                inverted_index[word].add(doc_id)

    save_index(index_dir, inverted_index)


def search_queries(data_dir, index_dir):
    results = SearchResults()
    queries_file = os.path.join(data_dir, 'vkmarco-doceval-queries.tsv')

    with copen(queries_file, 'r', encoding='utf-8') as queries_f:
        for line in queries_f:
            query_id, query_text = line.strip().split('\t', 1)
            query_id = int(query_id)
            matching_docs = None

            for token in preprocess(query_text):
                partition_key = int(sha224(token.encode("utf-8")).hexdigest(), 16) % 31
                index_file = Path(index_dir) / f"{partition_key}.json"

                if not index_file.exists():
                    matching_docs = set()
                    break

                with open(index_file, 'r') as f:
                    token_postings = set(json.load(f).get(token, []))

                matching_docs = token_postings if matching_docs is None else matching_docs.intersection(token_postings)

                if not matching_docs:
                    break

            results.add(query_id, matching_docs if matching_docs else set())

    return results


def main():
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='Output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='Force reindexing')
    parser.add_argument('--index_dir', required=True, help='Index directory')
    parser.add_argument('data_dir', help='Input data directory')
    args = parser.parse_args()

    start = timer()

    if args.build_index:
        build_index(args.data_dir, args.index_dir)
    else:
        results = search_queries(args.data_dir, args.index_dir)
        results.save_submission(
            objects_file=os.path.join(args.data_dir, 'objects.csv'),
            submission_file=args.submission_file
        )

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
