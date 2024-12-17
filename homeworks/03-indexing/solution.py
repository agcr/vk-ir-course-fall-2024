#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
from timeit import default_timer as timer

from nltk import tokenize
import pickle
import os


def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]


def display_progress(start, idx, doctype='documents'):
    partial_time = timer() - start
    print(f"\rProcessed {idx:5d} {doctype} in {partial_time:.3f}   ", end='', flush=True)


def build_index(docs, docsids):
    inverted_index = {}
    next_doc_id = 0
    for doc in docs:
        for word in doc:
            postings = inverted_index.setdefault(word, set())
            postings.add(docsids[next_doc_id])
        next_doc_id += 1
    return inverted_index


def search(query, inverted_index):
    results = None
    for word in query:
        postings = inverted_index.get(word, set())
        results = results & postings if results is not None else postings
    return results


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
        # Тут вы должны:
        # - загрузить тексты документов из файла args.data_dir/vkmarco-docs.tsv
        print("Opening docs file")
        docs_file = f'{args.data_dir}/vkmarco-docs.tsv'
        # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
        token_docs = []
        ids_docs = []
        docs_counter = 0
        with open(docs_file, encoding='utf-8') as docs:
            for line in docs.readlines():
                # we consider title and body (doc id, url, title, body)
                docid, _, title, body = line.rstrip('\n').split('\t')
                token_docs.append(preprocess(title + ' ' + body))
                docidx = int(docid[1:])
                ids_docs.append(docidx)
                # token_docs.append(preprocess(body))
                docs_counter += 1
                if docs_counter % 100 == 0:
                    display_progress(start, docs_counter, 'documents')
        display_progress(start, docs_counter, 'documents')
        # - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
        
        print("\nSaving inverted index doc file")
        docs_index = f'{args.index_dir}/docs_index.pkl'
        if not os.path.exists(args.index_dir):
            os.mkdir(args.index_dir)
        with open(docs_index, 'wb') as file:
            # file.write(str(build_index(token_docs)))
            pickle.dump(build_index(token_docs, ids_docs), file)
    else:
        print("Opening inverted index doc file")
        docs_index = f'{args.index_dir}/docs_index.pkl'
        with open(docs_index, 'rb') as file:
            # inverted_index = [f.rstrip() for f in file.readlines()]
            inverted_index = pickle.load(file)
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        queries_file = f'{args.data_dir}/vkmarco-doceval-queries.tsv'
        result = []
        query_idres = {}
        id_res_counter = 0
        with open(queries_file, encoding='utf-8') as queries:
            for query in queries.readlines():
                queryidx, query_text = query.rstrip().split('\t')
                # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
                preprocessed_query = preprocess(query_text)

                result.append(search(preprocessed_query, inverted_index))
                query_idres[int(queryidx)] = id_res_counter
                id_res_counter += 1
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена 
        # метка 1 (в документе есть все слова из запроса) или 0
        
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv

        print("\nGenerating submission file")
        obj_numerate = f'{args.data_dir}/objects.csv'
        # sample_submission_file = f'{args.data_dir}/sample_submission.csv'   # it seems the file is ordered so there is no need to upload
        submission_file = f'{args.data_dir}/submission.csv'
        total_res = 0
        with open(obj_numerate, encoding='utf-8') as objects, open(submission_file, 'w', encoding='utf-8') as fsub:
            next(objects)   # skip header
            fsub.write('ObjectId,Relevance\n')
            for obj in objects.readlines():
                obj_id, query_id, document_id = obj.rstrip().split(',')
                res = 1 if int(document_id[1:]) in result[query_idres[int(query_id)]] else 0
                total_res += res
                fsub.write(f'{obj_id},{res}\n')
        print(f"Total of {total_res} pairs queries/documents found.")

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
