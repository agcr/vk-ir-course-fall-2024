#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""
from argparse import ArgumentParser
from codecs import open as copen
from collections import defaultdict
from timeit import default_timer as timer
from pathlib import Path
from nltk import tokenize
from os import makedirs
from hashlib import sha224
from json import dump, load
from tqdm import tqdm


class Index:
    """
    Indexing homework solution
    """

    def __init__(self, index_file: str):
        self.i_index = {}
        with copen(index_file, mode='r', encoding='utf-8') as f:
            for line in f:
                index = line.split('\t')[0]
                header = line.split('\t')[2]
                text = line.split('\t')[3]
                for word in preprocess(text + header):
                    postings = self.i_index.setdefault(word, set())
                    postings.add(index)


class SearchResults:
    """
    Prints results of search into file
    """

    def __init__(self):
        self.results = set()

    def add(self, found: tuple[int, set]):
        query_index, documents_indices = found
        for document_index in documents_indices:
            self.results.add(f"{query_index}, {document_index}")

    def print_submission(self, objects_file: str, submission_file: str) -> None:
        with copen(submission_file, mode='w', encoding='utf-8') as submission_f:
            with copen(objects_file, mode='r', encoding='utf-8') as objects_f:
                for line in objects_f:
                    if line.startswith('ObjectId'):
                        submission_f.write("ObjectId,Relevance\n")
                        continue
                    ObjectId, QueryId, DocumentId = line.strip().split(',')
                    if f"{QueryId}, {DocumentId}" in self.results:
                        submission_f.write(f"{ObjectId},1\n")
                    else:
                        submission_f.write(f"{ObjectId},0\n")


def preprocess(text):
    """
    Preprocess text for indexing
    """
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def set_default(obj):
    """
    Set default values for objects
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def main():
    """
    Main function
    """
    # Парсим опции командной строки
    parser = ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    # Будем измерять время работы скрипта
    start = timer()

    # Какой у нас режим: построения индекса или генерации сабмишна?
    if args.build_index:
        idx = Index(f'{args.data_dir}/vkmarco-docs.tsv')

        makedirs(f'{args.index_dir}', exist_ok=True)
        result_dict = defaultdict(dict)
        for k, v in tqdm(idx.i_index.items()):
            h_name = int(sha224(k.encode("utf-8")).hexdigest(), 16) % 293
            result_dict[h_name].update({k: v})

        for k, v in result_dict.items():
            with open(f"{args.index_dir}/{k}.json", 'w+') as f:
                dump(v, f, default=set_default)

    else:
        search_results = SearchResults()
        with copen(f"{args.data_dir}/vkmarco-doceval-queries.tsv", mode='r', encoding='utf-8',
                   buffering=0) as queries_fh:
            for line in queries_fh:
                fields = line.rstrip('\n').split('\t')
                qid = int(fields[0])
                query = fields[1]
                for i, token in enumerate(preprocess(query)):
                    h_name = int(sha224(token.encode("utf-8")).hexdigest(), 16) % 293
                    if i == 0:
                        if Path(f"{args.index_dir}/{h_name}.json").exists():
                            with open(f"{args.index_dir}/{h_name}.json", 'r') as tmp_file:
                                valid_texts = set(load(tmp_file).get(token, set()))
                        else:
                            valid_texts = set()
                            break
                    else:
                        if Path(f"{args.index_dir}/{h_name}.json").exists():
                            with open(f"{args.index_dir}/{h_name}.json", 'r') as tmp_file:
                                texts = load(tmp_file).get(token, set())
                            valid_texts = valid_texts.intersection(set(texts))
                        else:
                            valid_texts = set()
                            break

                search_results.add((qid, valid_texts))

        search_results.print_submission(f"{args.data_dir}/objects.csv", args.submission_file)

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
