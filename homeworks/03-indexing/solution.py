#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
./solution.py --build_index --index_dir=index boolean-retrieval-homework-vk-ir-fall-2024

./solution.py --submission_file=submission.csv --index_dir=index boolean-retrieval-homework-vk-ir-fall-2024

systemd-run --quiet --user --scope -p MemoryMax=100M -p MemorySwapMax=0 ./solution.py --submission_file=submission.csv --index_dir=index boolean-retrieval-homework-vk-ir-fall-2024
"""

import argparse
from timeit import default_timer as timer
import os 
import pickle
import csv
import gc

MAX_DOCID = 1254065

import sys

import psutil  # Импортируем библиотеку psutil для мониторинга памяти

TOTAL_RELEVANT = []
EXPECTED_SYBMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789"

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Используемая память: {memory_info.rss / (1024 * 1024):.2f} MB")

def onle_expected_symbols(word):
    global EXPECTED_SYBMBOLS
    for char in word:
        if char not in EXPECTED_SYBMBOLS:
            return False
    return True

def preprocess(text):
    from nltk import tokenize

    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    return [token.lower() for token in tokens]

def transform_docid(docid):
    return int(docid[3:])

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


def write_index(path, index):
    # total_index = []
    print(f"Размер индекса в MB {sys.getsizeof(index) / (1024 * 1024)}")

    # for key in sorted(index):
    #     total_index.append([key, *sorted(index[key])])
    with open("test.pkl", "wb") as file:
        pickle.dump(index, file)
    # print(f"Размер преобразованного индекса в MB {sys.getsizeof(total_index) / (1024 * 1024)}")


def read_index(path):
    index = {}
    with open(path, "rb") as file:
        index = pickle.load(file)
    return index


def main():
    global TOTAL_RELEVANT
    parser = argparse.ArgumentParser(description='Indexing homework solution')
    parser.add_argument('--submission_file', help='output Kaggle submission file')
    parser.add_argument('--build_index', action='store_true', help='force reindexing')
    parser.add_argument('--index_dir', required=True, help='index directory')
    parser.add_argument('data_dir', help='input data directory')
    args = parser.parse_args()

    start = timer()
    # write_index("test.txt", {'c': [1, 2, 3], 'a': [5, 6], 'b': [10]})
    # print(read_index("test.txt"))
    # return
    print_memory_usage()
    if args.build_index:
        index = dict()
        with open(args.data_dir + "/vkmarco-docs.tsv", "r") as pages:
            for line in pages:
                fields = line.rstrip('\n').split('\t')
                docid, url, title, body = fields
                docid = transform_docid(docid)
                total_text = preprocess(title + " " + body)
                for word in total_text:
                    if not onle_expected_symbols(word):
                        continue
                    if word not in index:
                        index[word] = [docid]
                    else:
                        index[word].append(docid)
        gc.collect()
        if not os.path.exists(args.index_dir):
            os.makedirs(args.index_dir)
        print_memory_usage()
        write_index("test.pkl", index)

    else:
        # index = read_index("test.txt")
        index = {}
        print_memory_usage()
        with open("test.pkl", "rb") as file:
            index = pickle.load(file)
            gc.collect()
        # with open(args.index_dir + "/index.pkl", "rb") as file_index:
        #     index = pickle.load(file_index)
        print(f"Размер индекса в MB {sys.getsizeof(index) / (1024 * 1024)}")
        print_memory_usage()
        return 
        print_memory_usage()
        with open(args.data_dir + "/vkmarco-doceval-queries.tsv", "r") as querys:
            for line in querys:
                fields = line.rstrip('\n').split('\t')
                query_id, query_text = fields
                query_words = preprocess(query_text)
                relevant_docs = get_relevant_docs(query_words, index)
                for doc in relevant_docs:
                    TOTAL_RELEVANT.append((query_id, doc))
        print_memory_usage()
        result = []
        with open(args.data_dir + "/objects.csv", newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)            
            header = next(csv_reader)
            for row in csv_reader:
                ObjectId, QueryId, DocumentId = row[0], row[1], row[2]
                DocumentId = transform_docid(DocumentId)
                if (QueryId, DocumentId) in TOTAL_RELEVANT:
                    result.append((ObjectId, 1))
                else:
                    result.append((ObjectId, 0))
        print_memory_usage()
        with open(args.data_dir + "/" + args.submission_file, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['ObjectId', 'Label'])
            for row in result:
                csv_writer.writerow(row)

    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")
    print_memory_usage()

if __name__ == "__main__":
    main()
