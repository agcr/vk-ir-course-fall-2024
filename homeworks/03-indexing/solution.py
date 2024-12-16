#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
import pickle
import sys
import gc
import glob
import os
from timeit import default_timer as timer
from collections import defaultdict
from itertools import islice


def preprocess(text):
    from nltk import tokenize
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def build_index(docs_path):
    index_data = defaultdict(set)
    with open(os.path.join(docs_path, "vkmarco-docs.tsv"), 'r') as f:
        for line in f:
            doc_id, _, *doc_text_splitted = line.split("\t")
            lemmas = preprocess("\t".join(doc_text_splitted))
            for lemma in lemmas:
                index_data[lemma].add(doc_id)
    return index_data


def dict_chunks(data, size):
    """Функция взята из
    https://stackoverflow.com/questions/22878743/how-to-split-dictionary-into-multiple-dictionaries-fast"""
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def save_index(index: defaultdict[str, set], index_dir):
    def count_memory_entry(k: str, v: tuple):
        return sys.getsizeof(k) + sys.getsizeof(v) + sum((sys.getsizeof(e) for e in v)) + 50

    ans = {}
    cur_sz = sys.getsizeof(ans)
    for k in sorted(index.keys()):
        val = tuple(sorted(index[k]))
        entry_sz = count_memory_entry(k, val)
        cur_sz += entry_sz
        ans[k] = val
    parts_amount = cur_sz // (1024 * 1024 * 12)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    for ind, sub_index in enumerate(dict_chunks(ans, len(ans.keys()) // parts_amount)):
        with open(os.path.join(index_dir, f'index_{ind}.pickle'), 'wb') as f:
            pickle.dump(sub_index, f)


def intersect_lists(left, right):
    if left is None:
        return right
    ans = []
    first_it, second_it = 0, 0
    while first_it < len(left) and second_it < len(right):
        if left[first_it] < right[second_it]:
            first_it += 1
        elif right[second_it] < left[first_it]:
            second_it += 1
        else:
            ans.append(left[first_it])
            first_it += 1
            second_it += 1
    return ans[::]


def find_relevant_docs(index_path, query_lexemes) -> tuple:
    docs = None
    query_lexemes_copy = query_lexemes[::]
    for file_name in glob.glob(f"{index_path}/*.pickle"):
        with open(file_name, 'rb') as f:
            sub_ind = pickle.load(f)
        to_rem = []
        for l in query_lexemes_copy:
            if l in sub_ind:
                docs = intersect_lists(docs, sub_ind[l])
                to_rem.append(l)
                if not docs:
                    return ()
        for l in to_rem:
            query_lexemes_copy.remove(l)
        del sub_ind
        gc.collect()
    if query_lexemes_copy:
        return ()
    return tuple(docs)


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
        print("Indexing...")
        save_index(build_index(args.data_dir), args.index_dir)
        print(f"Index was saved in {args.index_dir}")
    else:
        print("Making submission...")
        queries = {}
        with open(os.path.join(args.data_dir, "vkmarco-doceval-queries.tsv"), 'r') as f:
            for line in f:
                k, *v = line.split('\t')
                queries[k] = preprocess('\t'.join(v))
        with open(args.submission_file, 'w') as sub_f:
            sub_f.write("ObjectId,Label\n")
            c = 0
            with open(os.path.join(args.data_dir, "objects.csv"), 'r') as obj_f:
                cached_query = None
                cur_docs = []
                skip_f = False
                for line in obj_f:
                    c += 1
                    if not skip_f:
                        skip_f = True
                        continue
                    obj_id, query_id, doc_id = line.rstrip().split(',')
                    if query_id != cached_query:
                        cur_docs = find_relevant_docs(args.index_dir, queries[query_id])
                        cached_query = query_id
                    sub_f.write(f"{obj_id},{1 if doc_id in cur_docs else 0}\n")
        print("Submission was made")

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
