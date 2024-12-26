#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indexing homework solution"""

import argparse
from collections import defaultdict
import csv
from hashlib import sha224
import os
from pathlib import Path
from timeit import default_timer as timer

from nltk import tokenize

def preprocess(text):
    # Tokenize
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Normalize
    return [token.lower() for token in tokens]

def build_index(data_dir):
    index = defaultdict(set)
    with open(os.path.join(data_dir, "vkmarco-docs.tsv")) as file:
        for line in file:
            doc_id, _, text = line.split('\t', 2)
            text = preprocess(text)
            for word in text:
                postings = index.setdefault(word, set())
                postings.add(doc_id)
        return index

def write_index_to_file(index, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    for term, ids in index.items():
        partition_id = int(sha224(term.encode("utf-8")).hexdigest(), 16) % 32
        with open(os.path.join(index_dir,  'index' + str(partition_id) + '.txt'), 'a') as file:
            file.write(term + "\t" + ','.join(ids) + "\n")

def read_tokens_from_file(index_path, word):
    with open(index_path, "r") as file:
        for row in file:
            term, ids = row.split('\t')
            if (word == term):
                return set(ids.strip().split(','))
    return set()

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
        index = build_index(args.data_dir)
        write_index_to_file(index, args.index_dir)
    else:
        # Тут вы должны:
        # - загрузить поисковые запросы из файла args.data_dir/vkmarco-doceval-queries.tsv
        # - прогнать запросы через индекс и, для каждого запроса, найти все документы, в которых есть все слова (термины) из запроса.
        # - для разбиения текстов запросов на слова тоже используем функцию preprocess()
        # - сформировать ваш сабмишн, в котором для каждого объекта (пары запрос-документ) будет проставлена метка 1 (в документе есть все слова из запроса) или 0
        #
        # Для формирования сабмишна надо загрузить и использовать файлы args.data_dir/sample_submission.csv и args.data_dir/objects.csv
        with open(os.path.join(args.data_dir, "vkmarco-doceval-queries.tsv")) as query_file:
            query_map = defaultdict(set)
            for line in query_file:
                query_id, query_text = line.split('\t')
                matching = set()
                for word in query_text.lower().split():
                    partition_id = int(sha224(word.encode("utf-8")).hexdigest(), 16) % 32
                    index_path = Path(os.path.join(args.index_dir, 'index' + str(partition_id) + '.txt'))
                    tokens = read_tokens_from_file(index_path, word)
                    matching = tokens if len(matching) == 0 else matching.intersection(tokens)
                    if len(matching) == 0:
                        break
                query_map[query_id] = matching


            with open(args.submission_file, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=['ObjectId', 'Label'])
                writer.writeheader()
                with open(os.path.join(args.data_dir, 'objects.csv'), "r") as objects:
                    reader = csv.reader(objects)
                    next(reader)
                    for object in reader:
                        object_id = object[0]
                        query_id = object[1]
                        document_id = object[2]
                        label = document_id in query_map[query_id]
                        writer.writerow({'ObjectId': object_id, 'Label': int(label)})

    # Репортим время работы скрипта
    elapsed = timer() - start
    print(f"finished, elapsed = {elapsed:.3f}")


if __name__ == "__main__":
    main()
