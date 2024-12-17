#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
import os
import shutil
import sys
from timeit import default_timer as timer

import psutil

CHECK_MEMORY = False


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


def check_memory():
    if not CHECK_MEMORY:
        return
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    if memory_mb > 100:  # 100MB limit
        print(f"Memory limit exceeded: {memory_mb:.2f} MB")
        sys.exit(1)
    return memory_mb


def get_deep_size(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen)
                    for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_size(item, seen) for item in obj)

    return size


print(f"Memory usage before: {get_memory_usage():.2f} MB")


def preprocess(text):
    from nltk import tokenize
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def save_segment(segment, index_dir, start_term, end_term):
    fname = f"{start_term}-{end_term}.tsv"

    with open(f"{index_dir}/{fname}", "w") as f:
        for term, doc_ids in segment.items():
            f.write(f"{term}\t{' '.join(doc_ids)}\n")


def segment_index(index, index_dir, max_size=10 * 1024 * 1024):
    sizes = {}
    for term, doc_ids in index.items():
        sizes[term] = get_deep_size(doc_ids)

    sizes = {k: v for k, v in sorted(sizes.items(), key=lambda item: item[0], reverse=True)}

    start_term = None
    end_term = None
    seg = 0
    segment = {}

    while len(sizes) > 0:
        term, size = sizes.popitem()

        if size > max_size:
            print(f"Term {term} is too large: {size / 1024 / 1024:.2f} MB")

        if seg + size >= max_size:
            save_segment(segment, index_dir, start_term, end_term)
            seg = 0
            start_term = None
            end_term = None
            segment = {}

        segment[term] = index.pop(term)
        seg += size

        if start_term is None:
            start_term = term

        end_term = term

    if len(segment) > 0:
        save_segment(segment, index_dir, start_term, end_term)


def build_index(data_dir, index_dir):
    index = {}
    with open(f"{data_dir}/vkmarco-docs.tsv") as f:
        for line in f:
            doc_id, fname, title, *text = line.strip().split("\t")
            terms = preprocess(title + " " + " ".join(text))
            for term in terms:
                index.setdefault(term, set()).add(doc_id)

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    else:
        shutil.rmtree(index_dir)
        os.makedirs(index_dir)

    segment_index(index, index_dir)


def load_terms(index_dir, terms):
    index = {}
    for term in terms:
        for fname in os.listdir(index_dir):
            start, end = fname.split(".")[0].split("-")

            if start <= term <= end:
                with open(f"{index_dir}/{fname}") as f:
                    for line in f:
                        term_, doc_ids = line.strip().split("\t")
                        check_memory()

                        if term_ == term:
                            index[term] = set(doc_ids.split(" "))
                            break

    return index


def make_submission(args):
    terms = set()
    queries = {}

    with open(f"{args.data_dir}/vkmarco-doceval-queries.tsv") as f:
        for line in f:
            query_id, query = line.strip().split("\t")
            query_terms = query.split(" ")
            terms.update(query_terms)
            queries[query_id] = query_terms
            check_memory()

        print(f"1. Current memory usage: {get_memory_usage():.2f} MB")

    print(f"2. Current memory usage: {get_memory_usage():.2f} MB")
    print(f"Terms size: {get_deep_size(terms) / 1024 / 1024:.2f} MB")
    print(f"Queries size: {get_deep_size(queries) / 1024 / 1024:.2f} MB")

    index = load_terms(args.index_dir, terms)

    print(f"3. Current memory usage: {get_memory_usage():.2f} MB")
    print(f"4. Index size: {get_deep_size(index) / 1024 / 1024:.2f} MB")

    del terms

    query_idx_to_doc_idx = {}
    for query_id, query_terms in queries.items():
        sets = []
        for term in query_terms:
            doc_ids = index.get(term, set())
            sets.append(doc_ids)
            check_memory()

        query_idx_to_doc_idx[query_id] = set.intersection(*sets)

    print(f"5. Current memory usage: {get_memory_usage():.2f} MB")
    print(f"6. Query index size: {get_deep_size(query_idx_to_doc_idx) / 1024 / 1024:.2f} MB")

    del index

    if os.path.exists(args.submission_file):
        os.remove(args.submission_file)

    with open(args.submission_file, "a") as submission_f:
        submission_f.write("ObjectId,Label\n")

        with open(f"{args.data_dir}/objects.csv") as objects_f:
            next(objects_f)

            for line in objects_f:
                obj_id, query_id, doc_id = line.strip().split(",")
                submission_f.write(f"{obj_id},{int(doc_id in query_idx_to_doc_idx[query_id])}\n")
                check_memory()

            print(f"7. Current memory usage: {get_memory_usage():.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()

    if args.build_index:
        build_index(args.data_dir, args.index_dir)

    else:
        make_submission(args)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
