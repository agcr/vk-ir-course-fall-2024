#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""
 
import argparse
from collections import defaultdict
import csv
from hashlib import sha224
import os
from timeit import default_timer as timer
 
from nltk import tokenize
 
 
 
def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
 
    # Normalize
    return [token.lower() for token in tokens]
 
def build_index_path(dir_path, word):
    return os.path.join(dir_path, 'inverted_index' +
                                        str(int(sha224(word.encode("utf-8")).hexdigest(), 16) % 32) + '.txt')
 
def build_index(data_dir, index_dir):
    docs_file = os.path.join(data_dir, 'vkmarco-docs.tsv')
    index = defaultdict(set)
    os.makedirs(index_dir, exist_ok=True)
 
    with open(docs_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc_id, _, text = line.strip().split('\t', 2)
            doc_id = doc_id
            terms = preprocess(text)
            for term in terms:
                index[term].add(doc_id)
        for term, doc_ids in index.items():
            index_file = build_index_path(index_dir, term)
            with open(index_file, 'a', encoding='utf-8') as out:
                out.write(f"{term}\t{','.join(map(str, doc_ids))}\n")
 
 
 
def load_tokens(index_file, word):
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            term, doc_ids = line.strip().split('\t')
            if (term == word):
                return set(doc_ids.split(','))
    return set()
 
def process_queries(data_dir, index_dir, submission_file):
    queries_file = os.path.join(data_dir, 'vkmarco-doceval-queries.tsv')
    objects_file = os.path.join(data_dir, 'objects.csv')
 
    with open(queries_file, 'r', encoding='utf-8') as queries_file:
        queries = {}
        for line in queries_file:
            query_id, query = line.strip().split('\t')
            query_id = int(query_id)
            terms = query.lower().split()
            result = None
            for term in terms:
                tokens = load_tokens(build_index_path(index_dir, term), term)
                if (result == None):
                    result = tokens
                else:
                    result = result.intersection(tokens)
                if len(result) == 0:
                    break
            queries[query_id] = result
 
 
 
        with open(objects_file, 'r', encoding='utf-8') as f, open(submission_file, 'w', newline='', encoding='utf-8') as out:
            reader = csv.reader(f)
            writer = csv.writer(out)
            writer.writerow(['ObjectId', 'Label'])
 
            next(reader)
            for row in reader:
                obj_id = row[0]
                query_id = row[1]
                doc_id = row[2]
                query = queries[int(query_id)]
                writer.writerow([obj_id, int(doc_id in queries[int(query_id)])])
 
 
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
        # - проиндексировать эти тексты, причем для разбиения текстов на слова (термины) надо воспользоваться функцией preprocess()
        # - сохранить получивший обратный индекс в папку переданную через параметр args.index_dir
        build_index(args.data_dir, args.index_dir)
    else:
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        #
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv
          process_queries(args.data_dir, args.index_dir, args.submission_file)
 
    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")
 
 
if __name__ == "__main__":
    main()
