#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""
import os
import argparse

import pandas as pd

from nltk import tokenize
from timeit import default_timer as timer

def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def build_index(doc_ids, texts):
    index = dict()
    for doc_id, text in zip(doc_ids, texts):
        for word in text:
            word_set = index.setdefault(word, set())
            word_set.add(doc_id)

    return index


def write_index(index, filepath, sep=','):
    lines = []

    for term in sorted(index):
        line = term + sep + sep.join(index[term]) + '\n'
        lines.append(line)
        
    with open(filepath, 'w') as f:
        f.writelines(lines)


def read_index(filepath, sep=','):
    index = dict()
    with open(filepath, 'r') as f:
        for line in f:
            splitted_line = line.strip().split(sep)
            index[splitted_line[0]] = set(splitted_line[1:])
            
    return index


def search(query_ids, queries, index):
    query_id2doc_id = dict()

    for query_id, query in zip(query_ids, queries):
        doc_ids = None
        for term in query:
            temp_doc_ids = index.get(term, set())
            doc_ids = doc_ids & temp_doc_ids if doc_ids is not None else temp_doc_ids 
    
        query_id2doc_id[query_id] = doc_ids

    return query_id2doc_id


def classify(query_ids, doc_ids, query_id2doc_id):
    result = []
    for i in range(len(query_ids)):
        result.append(int(doc_ids[i] in query_id2doc_id[query_ids[i]]))

    return result


def write_submission(object_ids, result, filepath):
    pd.DataFrame({'ObjectId': object_ids, 'Label': result}).to_csv(filepath, index=False)


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
        vkmarco_docs_path = os.path.join(args.data_dir, 'vkmarco-docs.tsv')
        vkmacro_docs = pd.read_csv(vkmarco_docs_path, header=None, names=['DocumentId', 'URL', 'TITLE', 'BODY'], sep='\t')
        vkmacro_docs['Text'] = vkmacro_docs[['TITLE', 'BODY']].apply(lambda x: str(x['TITLE']) + ' ' + str(x['BODY']), axis=1)
        vkmacro_docs = vkmacro_docs.drop(['URL', 'TITLE', 'BODY'], axis=1)
        vkmacro_docs['Text'] = vkmacro_docs['Text'].apply(preprocess)

        index = build_index(vkmacro_docs['DocumentId'].to_list(), vkmacro_docs['Text'].to_list())

        index_path = os.path.join(args.index_dir, 'index.txt')
        write_index(index, index_path, sep=',')
    else:
        vkmarco_doceval_queries_path = os.path.join(args.data_dir, 'vkmarco-doceval-queries.tsv')
        vkmarco_doceval_queries = pd.read_csv(vkmarco_doceval_queries_path, header=None, names=['QueryId', 'Query'], sep='\t')
        vkmarco_doceval_queries['Query'] = vkmarco_doceval_queries['Query'].apply(preprocess)

        objects_path = os.path.join(args.data_dir, 'objects.csv')
        objects = pd.read_csv(objects_path)

        index_path = os.path.join(args.index_dir, 'index.txt')
        index = read_index(index_path)
    
        query_id2doc_id = search(vkmarco_doceval_queries['QueryId'].tolist(), vkmarco_doceval_queries['Query'].tolist(), index)

        result = classify(objects['QueryId'].tolist(), objects['DocumentId'].tolist(), query_id2doc_id)
        write_submission(objects['ObjectId'].tolist(), result, args.submission_file)
    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
