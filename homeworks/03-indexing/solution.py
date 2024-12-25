#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import os
import glob
import pickle
import shutil
import argparse
import pandas as pd
from collections import defaultdict
from nltk import tokenize
from timeit import default_timer as timer

def preprocess(text):
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]

def reset_directory(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def build_inverted_index(docs_path: str, index_dir: str):
    index_part_size = 16
    word_to_file_parts = 52

    docs_file = os.path.join(docs_path, 'vkmarco-docs.tsv')
    docs_data = pd.read_csv(
        docs_file, sep='\t',
        names=['doc_id', 'url', 'title', 'text'],
        dtype={'doc_id': str, 'url': str, 'title': str, 'text': str},
        keep_default_na=False, na_values=['']
    )

    combined_texts = (docs_data['title'] + ' ' + docs_data['text']).values
    doc_ids = docs_data['doc_id'].values
    tokenized = [preprocess(txt) for txt in combined_texts]

    inv_index = defaultdict(set)
    for idx, doc_id in enumerate(doc_ids):
        for token in tokenized[idx]:
            inv_index[token].add(doc_id)

    index_chunks_dir = os.path.join(index_dir, 'index_chunks')
    reset_directory(index_chunks_dir)

    all_terms = list(inv_index.keys())
    term_file_map = {}

    for i in range(0, len(all_terms), index_part_size):
        batch = all_terms[i: i + index_part_size]
        batch_dict = {term: inv_index[term] for term in batch}
        file_number = (i // index_part_size) + 1
        term_file_map.update({term: file_number for term in batch})
        out_path = os.path.join(index_chunks_dir, f'chunk_{file_number}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(batch_dict, f)

    word_to_file_dir = os.path.join(index_dir, 'word_to_file')
    reset_directory(word_to_file_dir)
    term_items = list(term_file_map.items())
    chunk_size = max(len(term_items) // word_to_file_parts, 1)

    for i in range(0, len(term_items), chunk_size):
        submap = dict(term_items[i: i + chunk_size])
        submap_path = os.path.join(word_to_file_dir, f'word_to_file_{i // chunk_size + 1}.pkl')
        with open(submap_path, 'wb') as f:
            pickle.dump(submap, f)

def load_word_to_file(word_to_file_dir: str) -> list:
    return glob.glob(os.path.join(word_to_file_dir, '*.pkl'))

def get_file_index(term: str, word_to_file_paths: list) -> int | None:
    for path in word_to_file_paths:
        with open(path, 'rb') as f:
            submap = pickle.load(f)
        if term in submap:
            return submap[term]

def retrieve_documents(term: str, index_chunks_dir: str, word_to_file_paths: list) -> set:
    file_idx = get_file_index(term, word_to_file_paths)
    if file_idx is None:
        return set()

    target_file = os.path.join(index_chunks_dir, f'chunk_{file_idx}.pkl')
    with open(target_file, 'rb') as f:
        local_index = pickle.load(f)
    return local_index.get(term, set())

def search_queries(query_terms: list, index_dir: str, word_to_file_paths: list) -> set:
    index_chunks_dir = os.path.join(index_dir, 'index_chunks')
    if not query_terms:
        return set()

    common_docs = retrieve_documents(query_terms[0], index_chunks_dir, word_to_file_paths)
    for term in query_terms[1:]:
        current_docs = retrieve_documents(term, index_chunks_dir, word_to_file_paths)
        common_docs &= current_docs
        if not common_docs:
            break

    return common_docs

def generate_submission(index_dir: str, data_dir: str, submission_path: str):
    doc_batch_size = 9025
    word_to_file_paths = load_word_to_file(os.path.join(index_dir, 'word_to_file'))
    queries_file = os.path.join(data_dir, 'vkmarco-doceval-queries.tsv')
    objects_file = os.path.join(data_dir, 'objects.csv')

    queries_iter = pd.read_csv(
        queries_file, sep='\t',
        names=['query_id', 'query_text'],
        chunksize=1000
    )
    objects_iter = pd.read_csv(objects_file, chunksize=doc_batch_size)

    header_written = False

    for q_chunk in queries_iter:
        for _, row in q_chunk.iterrows():
            q_text = row['query_text']
            q_terms = preprocess(q_text)

            try:
                objs = next(objects_iter)
            except StopIteration:
                break

            matched_docs = search_queries(q_terms, index_dir, word_to_file_paths)
            result = pd.DataFrame({
                "ObjectId": objs["ObjectId"],
                "Label": objs["DocumentId"].isin(matched_docs).astype(int)
            })

            result.to_csv(submission_path, mode='a' if header_written else 'w',
                         index=False, header=not header_written)
            header_written = True

def main():
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()

    if args.build_index:
        build_inverted_index(args.data_dir, args.index_dir)
    else:
        generate_submission(args.index_dir, args.data_dir, args.submission_file)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")

if __name__ == "__main__":
    main()
